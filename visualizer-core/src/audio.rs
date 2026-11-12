use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, Sender};
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use wasapi::*;

use crate::config::{
    AudioSettings, BeatFocusMode, ChannelMixMode, DecayMode, FrequencyScaling,
    WeightingCurve, WindowFunction,
};

const MIN_BEAT_GAP_S: f64 = 0.25;   // max 240 BPM raw trigger rate
const BPM_FOLD_MIN: f64 = 70.0;
const BPM_FOLD_MAX: f64 = 180.0;
const BEAT_MEMORY_S: f64 = 8.0;     // forget beats older than this
const BEAT_TIMEOUT_S: f64 = 5.0;    // no beat for this long => BPM = 0
const MAX_BEATS: usize = 12;

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct BarFrame {
    pub values: Vec<f32>,
    pub peaks: Vec<f32>,
    pub beat: bool,
    pub beat_envelope: f32,
    pub bpm: f32,
    pub phase: f32,
}

pub struct AudioEngine {
    pub device_name: String,
    pub frame_rx: Receiver<BarFrame>,
    pub settings: Arc<ArcSwap<AudioSettings>>,
}

impl AudioEngine {
    pub fn new<F>(initial: AudioSettings, on_frame: F) -> Self
    where
        F: Fn() + Send + 'static,
    {
        let settings = Arc::new(ArcSwap::from_pointee(initial));

        let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
        let (sample_tx, sample_rx) = crossbeam_channel::bounded::<Vec<f32>>(32);
        let (info_tx, info_rx) = crossbeam_channel::bounded(1);

        let settings_for_capture = settings.clone();
        let capture_handle = thread::spawn(move || {
            capture_loop(settings_for_capture, sample_tx, info_tx);
        });

        let (device_name, sample_rate) = info_rx
            .recv()
            .unwrap_or_else(|_| ("Unknown".to_string(), 48000.0));

        let settings_for_analysis = settings.clone();
        thread::spawn(move || {
            analysis_loop(settings_for_analysis, sample_rx, frame_tx, sample_rate, on_frame);
        });

        drop(capture_handle);
        Self { device_name, frame_rx, settings }
    }
}

// ─────────────────────────────────────────────────────────────
// Capture thread — unchanged (already event-driven)
// ─────────────────────────────────────────────────────────────

fn capture_loop(
    settings: Arc<ArcSwap<AudioSettings>>,
    sample_tx: Sender<Vec<f32>>,
    info_tx: Sender<(String, f32)>,
) {
    let _ = initialize_mta();

    let device = match get_default_device(&Direction::Render) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("WASAPI: failed to get default render device: {e:?}");
            let _ = info_tx.send(("Unknown".to_string(), 48000.0));
            return;
        }
    };
    let mut audio_client = match device.get_iaudioclient() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("WASAPI: failed to get IAudioClient: {e:?}");
            let _ = info_tx.send(("Unknown".to_string(), 48000.0));
            return;
        }
    };

    let device_name = device.get_friendlyname().unwrap_or_else(|_| "Default Device".to_string());
    let format = match audio_client.get_mixformat() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WASAPI: failed to get mix format: {e:?}");
            let _ = info_tx.send((device_name, 48000.0));
            return;
        }
    };
    let sample_rate = format.get_samplespersec() as f32;
    let channels = format.get_nchannels() as usize;
    let blockalign = format.get_blockalign() as usize;
    let samples_per_frame = (blockalign / 4).max(1);

    let (_, min_time) = match audio_client.get_periods() {
        Ok(p) => p,
        Err(_) => (100_000i64, 100_000i64),
    };
    if audio_client.initialize_client(&format, min_time, &Direction::Capture, &ShareMode::Shared, true).is_err() {
        let _ = info_tx.send((device_name, sample_rate));
        return;
    }

    let h_event = match audio_client.set_get_eventhandle() {
        Ok(h) => h,
        Err(_) => { let _ = info_tx.send((device_name, sample_rate)); return; }
    };
    let capture_client = match audio_client.get_audiocaptureclient() {
        Ok(c) => c,
        Err(_) => { let _ = info_tx.send((device_name, sample_rate)); return; }
    };

    if audio_client.start_stream().is_err() {
        let _ = info_tx.send((device_name, sample_rate));
        return;
    }
    let _ = info_tx.send((device_name, sample_rate));

    let mut sample_buffer: Vec<f32> = Vec::new();
    let mut mono: Vec<f32> = Vec::with_capacity(4096);

    loop {
        let waited_at = Instant::now();
        if h_event.wait_for_event(250).is_err() && waited_at.elapsed() < Duration::from_millis(20) {
            thread::sleep(Duration::from_millis(100));
        }

        loop {
            let packet_size = capture_client.get_next_nbr_frames().ok().flatten().unwrap_or(0);
            if packet_size == 0 { break; }

            let needed = packet_size as usize * samples_per_frame;
            sample_buffer.resize(needed, 0.0);
            let byte_slice: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    sample_buffer.as_mut_ptr() as *mut u8,
                    needed * std::mem::size_of::<f32>(),
                )
            };
            let (frames_read, _) = capture_client.read_from_device(byte_slice).unwrap_or((0, BufferFlags::none()));
            if frames_read == 0 { break; }

            let valid_samples = (frames_read as usize * samples_per_frame).min(sample_buffer.len());
            let samples = &sample_buffer[..valid_samples];

            let mix_mode = settings.load().channel_mix;
            mono.clear();
            mono.reserve(valid_samples / channels.max(1));
            for chunk in samples.chunks_exact(channels.max(1)) {
                if channels >= 2 {
                    let l = chunk[0];
                    let r = chunk[1];
                    let val = match mix_mode {
                        ChannelMixMode::Mono | ChannelMixMode::Mid => (l + r) * 0.5,
                        ChannelMixMode::Left => l,
                        ChannelMixMode::Right => r,
                        ChannelMixMode::Side => (l - r) * 0.5,
                    };
                    mono.push(val);
                } else {
                    mono.push(chunk[0]);
                }
            }
            if mono.is_empty() { continue; }
            let out = std::mem::replace(&mut mono, Vec::with_capacity(4096));
            if sample_tx.send(out).is_err() { return; }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Analysis thread
// ─────────────────────────────────────────────────────────────

const LOUD_THRESHOLD: f32 = 1.0e-4;
const IDLE_AFTER: Duration = Duration::from_millis(700);

fn peak_abs(chunk: &[f32]) -> f32 {
    chunk.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

struct DcBlocker { alpha: f32, x: f32, y: f32 }
impl DcBlocker {
    fn new(sample_rate: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * 10.0);
        let dt = 1.0 / sample_rate.max(1.0);
        Self { alpha: rc / (rc + dt), x: 0.0, y: 0.0 }
    }
    fn apply(&mut self, chunk: &mut [f32]) {
        for s in chunk.iter_mut() {
            let y = self.alpha * (self.y + *s - self.x);
            self.x = *s;
            self.y = y;
            *s = y;
        }
    }
}

fn analysis_loop<F: Fn() + Send + 'static>(
    settings: Arc<ArcSwap<AudioSettings>>,
    sample_rx: Receiver<Vec<f32>>,
    frame_tx: Sender<BarFrame>,
    sample_rate: f32,
    on_frame: F,
) {
    let mut processor = AudioProcessor::new(sample_rate);
    // Pre-allocated history buffer; we reuse it instead of reallocating.
    let mut history: Vec<f32> = Vec::with_capacity(16_384);
    let mut dc = DcBlocker::new(sample_rate);
    let sleeper = spin_sleep::SpinSleeper::new(500_000);

    let mut last_tick = Instant::now();
    let mut last_loud = Instant::now();
    let mut last_chunk = Instant::now();
    let mut idle = false;
    let mut last_sent_zero = false;

    loop {
        let current = settings.load_full();

        if idle {
            let mut chunk = match sample_rx.recv() {
                Ok(c) => c,
                Err(_) => return,
            };
            if peak_abs(&chunk) <= LOUD_THRESHOLD {
                continue;
            }
            idle = false;
            let now = Instant::now();
            last_loud = now;
            last_chunk = now;
            last_tick = now;
            if current.dc_block { dc.apply(&mut chunk); }
            history.extend_from_slice(&chunk);
        }

        let poll_period = Duration::from_secs_f32(1.0 / current.poll_rate_hz.max(1.0));

        let mut got_chunk = false;
        while let Ok(mut chunk) = sample_rx.try_recv() {
            got_chunk = true;
            if peak_abs(&chunk) > LOUD_THRESHOLD { last_loud = Instant::now(); }
            if current.dc_block { dc.apply(&mut chunk); }
            history.extend_from_slice(&chunk);
        }
        if got_chunk { last_chunk = Instant::now(); }

        let dt_raw = last_tick.elapsed().as_secs_f32();
        let dt = dt_raw.clamp(0.001, 0.2);
        last_tick = Instant::now();

        if !got_chunk && last_chunk.elapsed() > Duration::from_millis(60) {
            let n = ((dt_raw * sample_rate) as usize).min(16_384);
            // Reuse capacity: resize only if needed.
            history.resize(history.len() + n, 0.0);
        }

        let cap = (current.fft_size * 4).max(8192);
        if history.len() > cap {
            let drop_n = history.len() - cap;
            history.drain(0..drop_n);
        }

        if let Some(frame) = processor.process(&history, &current, dt) {
            let is_zero = frame.values.iter().all(|&v| v == 0.0)
                && frame.peaks.iter().all(|&v| v == 0.0);
            if !(is_zero && last_sent_zero) {
                if frame_tx.try_send(frame).is_ok() {
                    last_sent_zero = is_zero;
                    on_frame();
                }
            }
        }

        if last_loud.elapsed() > IDLE_AFTER && processor.is_silent() {
            processor.snap_to_silence();
            let frame = processor.silent_frame();
            on_frame();
            let _ = frame_tx.send_timeout(frame, Duration::from_millis(200));
            on_frame();
            last_sent_zero = true;
            history.clear();
            history.resize(current.fft_size, 0.0);
            idle = true;
            continue;
        }

        let elapsed = last_tick.elapsed();
        if elapsed < poll_period {
            sleeper.sleep(poll_period - elapsed);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// The processor — buffers are pre-allocated and reused.
// ─────────────────────────────────────────────────────────────

struct AudioProcessor {
    sample_rate: f32,
    cached_fft_size: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    cached_window_fn: WindowFunction,
    input_buffer: Vec<f32>,
    output_buffer: Vec<Complex32>,
    scratch_buffer: Vec<Complex32>,

    cached_num_bars: usize,
    cached_scaling: FrequencyScaling,
    cached_min_freq: f32,
    cached_max_freq: f32,
    bin_ranges: Vec<(usize, usize)>,
    bin_centers_hz: Vec<f32>,

    bar_values: Vec<f32>,
    bar_peaks: Vec<f32>,
    bar_velocities: Vec<f32>,
    peak_hold_remaining: Vec<f32>,

    // Reusable scratch buffers (sized to num_bars)
    raw_bars: Vec<f32>,
    tmp_bars: Vec<f32>,
    bloom_bars: Vec<f32>,
    mapped: Vec<f32>,
    smoothed: Vec<f32>,

    // Flat ring buffer for frame averaging: `avg * num_bars` values.
    frame_ring: Vec<f32>,
    frame_ring_pos: usize,
    frame_ring_count: usize,

    agc_envelope: f32,

    // Beat history: VecDeque + running sum.
    beat_energy_history: VecDeque<f32>,
    beat_energy_sum: f32,
    beat_times: VecDeque<f64>,
    bpm: f32,
    phase: f32,
    beat_envelope: f32,
    time_elapsed: f64,

    // Patch 6: edge-detection and cached per-bar dB offsets.
    beat_armed: bool,
    bar_offset_db: Vec<f32>,
    offset_centers: Vec<f32>,
    offset_params: [f32; 4],
    offset_weighting: u8,
}

impl AudioProcessor {
    fn new(sample_rate: f32) -> Self {
        let mut planner = RealFftPlanner::new();
        let fft_size = 2048;
        let window_fn = WindowFunction::Hann;
        let fft = planner.plan_fft_forward(fft_size);
        let input_buffer = fft.make_input_vec();
        let output_buffer = fft.make_output_vec();
        let scratch_buffer = fft.make_scratch_vec();

        let window = make_window(window_fn, fft_size);
        let bin_ranges = make_bin_ranges(fft_size, 80, FrequencyScaling::Logarithmic,
                                          sample_rate, 30.0, 16000.0);
        let bin_centers_hz = centers_from_ranges(&bin_ranges, fft_size, sample_rate);

        let num_bars = 80;
        let avg = 1;

        Self {
            sample_rate,
            cached_fft_size: fft_size,
            fft,
            window,
            cached_window_fn: window_fn,
            input_buffer,
            output_buffer,
            scratch_buffer,
            cached_num_bars: num_bars,
            cached_scaling: FrequencyScaling::Logarithmic,
            cached_min_freq: 30.0,
            cached_max_freq: 16000.0,
            bin_ranges,
            bin_centers_hz,
            bar_values: vec![0.0; num_bars],
            bar_peaks: vec![0.0; num_bars],
            bar_velocities: vec![0.0; num_bars],
            peak_hold_remaining: vec![0.0; num_bars],
            raw_bars: vec![0.0; num_bars],
            tmp_bars: vec![0.0; num_bars],
            bloom_bars: vec![0.0; num_bars],
            mapped: vec![0.0; num_bars],
            smoothed: vec![0.0; num_bars],
            frame_ring: vec![0.0; avg * num_bars],
            frame_ring_pos: 0,
            frame_ring_count: 0,
            agc_envelope: 1.0,
            beat_energy_history: VecDeque::with_capacity(240),
            beat_energy_sum: 0.0,
            beat_times: VecDeque::with_capacity(8),
            bpm: 0.0,
            phase: 0.0,
            beat_envelope: 0.0,
            time_elapsed: 0.0,
            beat_armed: true,
            bar_offset_db: Vec::new(),
            offset_centers: Vec::new(),
            offset_params: [f32::NAN; 4], // NaN never equals => forces first build
            offset_weighting: u8::MAX,
        }
    }

    /// tilt + EQ + weighting depend only on settings and the bar's centre
    /// frequency, so compute them when those change, not every frame.
    fn refresh_bar_offsets(&mut self, s: &AudioSettings) {
        let w_id = match s.weighting {
            WeightingCurve::None => 0u8,
            WeightingCurve::AWeighting => 1,
            WeightingCurve::KWeighting => 2,
        };
        let params = [s.spectrum_tilt, s.eq_bass_db, s.eq_mid_db, s.eq_treble_db];
        if self.offset_params == params
            && self.offset_weighting == w_id
            && self.offset_centers == self.bin_centers_hz
        {
            return;
        }
        self.offset_params = params;
        self.offset_weighting = w_id;
        self.offset_centers.clear();
        self.offset_centers.extend_from_slice(&self.bin_centers_hz);

        self.bar_offset_db.clear();
        for &f in &self.bin_centers_hz {
            let tilt_db = s.spectrum_tilt * (f / 1000.0).log2();
            let eq_db = if f < 250.0 {
                s.eq_bass_db
            } else if f < 4000.0 {
                s.eq_mid_db
            } else {
                s.eq_treble_db
            };
            let weight_db = match s.weighting {
                WeightingCurve::None => 0.0,
                WeightingCurve::AWeighting => a_weighting_db(f),
                WeightingCurve::KWeighting => k_weighting_db(f),
            };
            self.bar_offset_db.push(tilt_db + eq_db + weight_db);
        }
    }

    fn update_initial_state(&mut self, s: &AudioSettings) {
        if s.fft_size != self.cached_fft_size {
            let mut planner = RealFftPlanner::new();
            self.fft = planner.plan_fft_forward(s.fft_size);
            self.input_buffer = self.fft.make_input_vec();
            self.output_buffer = self.fft.make_output_vec();
            self.scratch_buffer = self.fft.make_scratch_vec();
            self.cached_fft_size = s.fft_size;
            self.cached_window_fn = s.window_function;
            self.window = make_window(s.window_function, s.fft_size);
            self.bin_ranges = make_bin_ranges(s.fft_size, s.num_bars, s.frequency_scaling,
                                              self.sample_rate, s.min_frequency_hz, s.max_frequency_hz);
            self.bin_centers_hz = centers_from_ranges(&self.bin_ranges, s.fft_size, self.sample_rate);
        }
        if s.window_function != self.cached_window_fn {
            self.cached_window_fn = s.window_function;
            self.window = make_window(s.window_function, s.fft_size);
        }
        if s.num_bars != self.cached_num_bars
            || s.frequency_scaling != self.cached_scaling
            || s.min_frequency_hz != self.cached_min_freq
            || s.max_frequency_hz != self.cached_max_freq
        {
            self.cached_num_bars = s.num_bars;
            self.cached_scaling = s.frequency_scaling;
            self.cached_min_freq = s.min_frequency_hz;
            self.cached_max_freq = s.max_frequency_hz;
            self.bin_ranges = make_bin_ranges(s.fft_size, s.num_bars, s.frequency_scaling,
                                              self.sample_rate, s.min_frequency_hz, s.max_frequency_hz);
            self.bin_centers_hz = centers_from_ranges(&self.bin_ranges, s.fft_size, self.sample_rate);
            self.resize_bar_state(s.num_bars);
        }
    }

    fn is_silent(&self) -> bool {
        self.bar_values.iter().all(|&v| v < 1.0e-3)
            && self.bar_peaks.iter().all(|&v| v < 1.0e-3)
            && self.beat_envelope < 1.0e-3
    }

    fn snap_to_silence(&mut self) {
        for v in self.bar_values.iter_mut() { *v = 0.0; }
        for v in self.bar_peaks.iter_mut() { *v = 0.0; }
        for v in self.bar_velocities.iter_mut() { *v = 0.0; }
        for v in self.peak_hold_remaining.iter_mut() { *v = 0.0; }
        self.frame_ring.fill(0.0);
        self.frame_ring_count = 0;
        self.beat_energy_history.clear();
        self.beat_energy_sum = 0.0;
        self.beat_envelope = 0.0;
        // New silence => the next track starts from scratch.
        self.beat_times.clear();
        self.beat_armed = true;
        self.bpm = 0.0;
    }

    fn silent_frame(&self) -> BarFrame {
        BarFrame {
            values: vec![0.0; self.bar_values.len()],
            peaks: vec![0.0; self.bar_peaks.len()],
            beat: false,
            beat_envelope: 0.0,
            bpm: self.bpm,
            phase: self.phase,
        }
    }

    fn resize_bar_state(&mut self, n: usize) {
        self.bar_values.resize(n, 0.0);
        self.bar_peaks.resize(n, 0.0);
        self.bar_velocities.resize(n, 0.0);
        self.peak_hold_remaining.resize(n, 0.0);
        self.raw_bars.resize(n, 0.0);
        self.tmp_bars.resize(n, 0.0);
        self.bloom_bars.resize(n, 0.0);
        self.mapped.resize(n, 0.0);
        self.smoothed.resize(n, 0.0);
        // Frame ring is resized lazily in process() based on avg.
        self.frame_ring_pos = 0;
        self.frame_ring_count = 0;
    }

    fn process(&mut self, history: &[f32], s: &AudioSettings, dt: f32) -> Option<BarFrame> {
        self.update_initial_state(s);
        self.time_elapsed += dt as f64;

        let n_fft = self.cached_fft_size;
        if history.len() < n_fft {
            return Some(BarFrame {
                values: self.bar_values.clone(),
                peaks: self.bar_peaks.clone(),
                beat: false,
                beat_envelope: self.beat_envelope,
                bpm: self.bpm,
                phase: self.phase,
            });
        }
        let window_slice = &history[history.len() - n_fft..];

        // AGC (before windowing)
        let raw_rms = {
            let sum_sq: f32 = window_slice.iter().map(|x| x * x).sum();
            (sum_sq / n_fft as f32).sqrt()
        };
        if s.agc_enabled {
            let desired = 0.08 / raw_rms.max(1e-6);
            let coeff = if desired < self.agc_envelope {
                s.agc_attack.clamp(0.0001, 1.0)
            } else {
                s.agc_release.clamp(0.0001, 1.0)
            };
            self.agc_envelope += (desired - self.agc_envelope) * coeff;
        } else {
            self.agc_envelope = 1.0;
        }
        let agc_gain = self.agc_envelope.clamp(0.01, 20.0);
        let user_gain = s.raw_gain.max(0.01) * agc_gain;

        self.input_buffer.copy_from_slice(window_slice);
        for i in 0..n_fft {
            self.input_buffer[i] *= self.window[i];
        }
        if self.fft.process_with_scratch(&mut self.input_buffer, &mut self.output_buffer, &mut self.scratch_buffer).is_err() {
            return None;
        }

        let num_bars = self.cached_num_bars;
        let nyquist = n_fft / 2;
        let norm = 1.0 / (n_fft as f32 / 2.0) * user_gain;

        // ── raw per-bar magnitudes (reuse buffer) ────────────────
        for i in 0..num_bars {
            let (start, end) = self.bin_ranges[i];
            let end = end.min(nyquist).max(start + 1);
            let mut mag = 0.0f32;
            for bin in start..end {
                mag += self.output_buffer[bin].norm();
            }
            mag /= (end - start) as f32;
            self.raw_bars[i] = (mag * norm + 1e-9).max(1e-9);
        }

        // ── frequency-domain smoothing (reuse tmp) ───────────────
        if s.frequency_smoothing > 0.0 {
            let alpha = s.frequency_smoothing.clamp(0.0, 1.0);
            self.tmp_bars.copy_from_slice(&self.raw_bars);
            for i in 0..num_bars {
                let lo = i.saturating_sub(1);
                let hi = (i + 1).min(num_bars - 1);
                let avg = (self.raw_bars[lo] + self.raw_bars[i] * 2.0 + self.raw_bars[hi]) * 0.25;
                self.tmp_bars[i] = self.raw_bars[i] * (1.0 - alpha) + avg * alpha;
            }
            self.raw_bars.copy_from_slice(&self.tmp_bars);
        }

        // ── harmonic emphasis (reuse bloom) ──────────────────────
        if s.harmonic_emphasis && num_bars >= 5 {
            self.bloom_bars.copy_from_slice(&self.raw_bars);
            for i in 2..num_bars - 2 {
                let c = self.raw_bars[i];
                let l1 = self.raw_bars[i - 1];
                let r1 = self.raw_bars[i + 1];
                let l2 = self.raw_bars[i - 2];
                let r2 = self.raw_bars[i + 2];
                if c > l1 && c > r1 && c > l2 && c > r2 && c > 0.005 {
                    let spread = c * 0.15;
                    self.bloom_bars[i - 1] = self.bloom_bars[i - 1].max(spread * 0.7);
                    self.bloom_bars[i + 1] = self.bloom_bars[i + 1].max(spread * 0.7);
                    self.bloom_bars[i - 2] = self.bloom_bars[i - 2].max(spread * 0.3);
                    self.bloom_bars[i + 2] = self.bloom_bars[i + 2].max(spread * 0.3);
                }
            }
            self.raw_bars.copy_from_slice(&self.bloom_bars);
        }

        // ── dB mapping (reuse mapped) ────────────────────────────
        self.refresh_bar_offsets(s);
        let sens_norm = (s.sensitivity / 100.0).clamp(0.0, 1.0);
        let min_db_effective = s.min_decibels - (sens_norm - 0.5) * 20.0;
        let db_span = (s.max_decibels - min_db_effective).max(1.0);
        let target_ratio = db_span / s.target_dynamic_range_db.max(1.0);
        let gamma = s.response_gamma.max(0.1) * target_ratio;

        for i in 0..num_bars {
            let db = 20.0 * self.raw_bars[i].log10();
            let tilted_db = db + self.bar_offset_db.get(i).copied().unwrap_or(0.0);
            let mut n01 = ((tilted_db - min_db_effective) / db_span).clamp(0.0, 1.0);
            if n01 < s.noise_gate {
                n01 = 0.0;
            }
            self.mapped[i] = n01.powf(gamma);
        }

        // ── time-domain smoothing using a flat ring buffer ───────
        let avg = s.average_size.max(1);
        let needed = avg * num_bars;
        if self.frame_ring.len() != needed {
            self.frame_ring.resize(needed, 0.0);
            self.frame_ring_pos = 0;
            self.frame_ring_count = 0;
        }
        if self.frame_ring_count < avg {
            self.frame_ring_count += 1;
        }
        let base = self.frame_ring_pos * num_bars;
        self.frame_ring[base..base + num_bars].copy_from_slice(&self.mapped);
        self.frame_ring_pos = (self.frame_ring_pos + 1) % avg;

        // Sum all frames in the ring (reuse `smoothed` as accumulator).
        self.smoothed.fill(0.0);
        for f in 0..self.frame_ring_count {
            let offset = f * num_bars;
            for i in 0..num_bars {
                self.smoothed[i] += self.frame_ring[offset + i];
            }
        }
        let inv = 1.0 / self.frame_ring_count as f32;
        for v in self.smoothed.iter_mut() { *v *= inv; }

        // ── per-frame peak normalization ──────────────────────────
        if s.per_frame_normalization {
            let peak = self.smoothed.iter().cloned().fold(0.0f32, f32::max);
            if peak > 0.001 {
                for v in self.smoothed.iter_mut() { *v /= peak; }
            }
        }

        // ── FIXED: move `smoothed` out to avoid the double-borrow ─
        let smoothed = std::mem::take(&mut self.smoothed);
        self.update_ballistics(&smoothed, s, dt);
        self.smoothed = smoothed;

        let (beat, phase) = self.detect_rhythm(s, dt);

        Some(BarFrame {
            values: self.bar_values.clone(),
            peaks: self.bar_peaks.clone(),
            beat,
            beat_envelope: self.beat_envelope,
            bpm: self.bpm,
            phase,
        })
    }

    fn update_ballistics(&mut self, inputs: &[f32], s: &AudioSettings, dt: f32) {
        let n = self.bar_values.len().min(inputs.len());
        let peak_hold_s = s.peak_hold_ms / 1000.0;

        for i in 0..n {
            let (atk_ms, dec_ms) = if s.per_band_ballistics {
                let f = self.bin_centers_hz.get(i).copied().unwrap_or(1000.0);
                if f < 250.0 {
                    (s.bass_attack_ms, s.bass_decay_ms)
                } else if f < 4000.0 {
                    (s.mid_attack_ms, s.mid_decay_ms)
                } else {
                    (s.treble_attack_ms, s.treble_decay_ms)
                }
            } else {
                (s.attack_time_ms, s.decay_time_ms)
            };

            let attack_alpha = if atk_ms <= 0.0 {
                1.0
            } else {
                1.0 - (-dt * 1000.0 / atk_ms).exp()
            };
            let decay_s = (dec_ms / 1000.0).max(0.001);

            let input = inputs[i].clamp(0.0, 1.0);
            let prev = self.bar_values[i];

            if input > prev {
                self.bar_velocities[i] = 0.0;
                self.bar_values[i] = prev + (input - prev) * attack_alpha;
            } else {
                self.bar_values[i] = match s.decay_mode {
                    DecayMode::Linear => (prev - dt / decay_s).max(input),
                    DecayMode::Exponential => {
                        let tau = decay_s / 3.0;
                        (input + (prev - input) * (-dt / tau).exp()).max(input)
                    }
                    DecayMode::Gravity => {
                        let g = 2.0 / (decay_s * decay_s) * s.gravity_coefficient.max(0.01);
                        self.bar_velocities[i] += g * dt;
                        (prev - self.bar_velocities[i] * dt).max(input)
                    }
                };
            }
            self.bar_values[i] = self.bar_values[i].clamp(0.0, 1.0);

            if peak_hold_s <= 0.0 {
                self.bar_peaks[i] = self.bar_values[i];
                self.peak_hold_remaining[i] = 0.0;
            } else if self.bar_values[i] >= self.bar_peaks[i] {
                self.bar_peaks[i] = self.bar_values[i];
                self.peak_hold_remaining[i] = peak_hold_s;
            } else if self.peak_hold_remaining[i] > 0.0 {
                self.peak_hold_remaining[i] -= dt;
            } else {
                let fall_rate = 1.0 / decay_s;
                self.bar_peaks[i] = (self.bar_peaks[i] - dt * fall_rate).max(self.bar_values[i]);
            }
        }
    }

    fn detect_rhythm(&mut self, s: &AudioSettings, dt: f32) -> (bool, f32) {
        let n_fft = self.cached_fft_size;
        let nyquist = n_fft / 2;
        let bin_width = self.sample_rate / n_fft as f32;

        // Band edges as bin indices (bass < 250 Hz <= mid < 4 kHz <= treble).
        let bass_end = ((250.0 / bin_width).ceil() as usize).clamp(1, nyquist);
        let mid_end = ((4000.0 / bin_width).ceil() as usize).clamp(bass_end, nyquist);
        let (lo, hi) = match s.beat_focus {
            BeatFocusMode::Bass => (1, bass_end),
            BeatFocusMode::Mid => (bass_end, mid_end),
            BeatFocusMode::Treble => (mid_end, nyquist),
        };
        // Only the focused band is summed (was: the whole spectrum, every frame).
        let energy: f32 = self.output_buffer[lo..hi.max(lo)].iter().map(|c| c.norm()).sum();

        // ~1.5 s sliding window of band energy.
        let hist_len = (s.poll_rate_hz * 1.5).max(30.0) as usize;
        while self.beat_energy_history.len() >= hist_len {
            self.beat_energy_history.pop_front();
        }
        self.beat_energy_history.push_back(energy);
        // Exact sum (<= a few hundred entries): no f32 drift like a running sum.
        self.beat_energy_sum = self.beat_energy_history.iter().sum();
        let local_avg = self.beat_energy_sum / self.beat_energy_history.len() as f32;

        let sensitivity = s.beat_detection_sensitivity.clamp(0.0, 1.0);
        let threshold =
            local_avg * s.energy_threshold_multiplier.max(1.0) * (1.0 - sensitivity * 0.5);
        let above = energy > threshold && energy > local_avg * 1.05;

        // Hysteresis: re-arm only after the energy has dropped back down.
        if energy < threshold * 0.9 {
            self.beat_armed = true;
        }
        let since_last = self.time_elapsed
            - self.beat_times.back().copied().unwrap_or(f64::NEG_INFINITY);
        let is_beat = above && self.beat_armed && since_last > MIN_BEAT_GAP_S;
        if above {
            // Consume the crossing, even if the lockout rejected it,
            // so a sustained loud passage can't re-trigger later.
            self.beat_armed = false;
        }

        if is_beat {
            self.beat_times.push_back(self.time_elapsed);
            while self.beat_times.len() > MAX_BEATS {
                self.beat_times.pop_front();
            }
            self.recompute_bpm();
        }

        // Expire stale beats / stale tempo.
        while let Some(&t) = self.beat_times.front() {
            if self.time_elapsed - t > BEAT_MEMORY_S {
                self.beat_times.pop_front();
            } else {
                break;
            }
        }
        if let Some(&last) = self.beat_times.back() {
            if self.time_elapsed - last > BEAT_TIMEOUT_S {
                self.beat_times.clear();
                self.bpm = 0.0;
            }
        }

        if is_beat {
            self.beat_envelope = 1.0;
        } else if self.beat_envelope > 0.0 {
            let tau = (s.beat_envelope_decay_ms / 1000.0).max(0.001);
            self.beat_envelope *= (-dt / tau).exp();
            if self.beat_envelope < 0.001 {
                self.beat_envelope = 0.0;
            }
        }

        let period_s = if self.bpm > 20.0 { 60.0 / self.bpm } else { 0.5 };
        self.phase = (self.phase + dt / period_s).rem_euclid(1.0);

        if is_beat && s.bpm_lock_enabled {
            let gain = s.bpm_phase_smoothing.clamp(0.0, 1.0);
            let error = if self.phase < 0.5 { -self.phase } else { 1.0 - self.phase };
            self.phase = (self.phase + error * gain).rem_euclid(1.0);
        }

        (is_beat, self.phase)
    }

    fn recompute_bpm(&mut self) {
        // Need at least 2 intervals (3 beats) before trusting anything.
        if self.beat_times.len() < 3 {
            return;
        }
        let min_iv = 60.0 / BPM_FOLD_MAX;
        let max_iv = 60.0 / BPM_FOLD_MIN;

        // Fold every interval into [min_iv, max_iv]: a missed beat (2x) or a
        // double trigger (0.5x) then votes for the same tempo as a normal one.
        let mut iv: Vec<f64> = self
            .beat_times
            .iter()
            .zip(self.beat_times.iter().skip(1))
            .map(|(a, b)| b - a)
            .filter(|d| *d > 0.05)
            .map(|mut d| {
                while d < min_iv { d *= 2.0; }
                while d > max_iv { d *= 0.5; }
                d
            })
            .collect();
        if iv.len() < 2 {
            return;
        }
        iv.sort_by(|a, b| a.total_cmp(b));
        let median = iv[iv.len() / 2];

        // Average only the intervals that agree with the median (reject outliers).
        let (sum, n) = iv
            .iter()
            .filter(|d| (**d - median).abs() <= median * 0.12)
            .fold((0.0f64, 0usize), |(s, n), d| (s + d, n + 1));
        if n == 0 {
            return;
        }
        let new_bpm = (60.0 / (sum / n as f64)) as f32;

        if self.bpm <= 0.0 || (new_bpm - self.bpm).abs() / self.bpm > 0.12 {
            self.bpm = new_bpm; // first estimate or genuine tempo change: jump
        } else {
            self.bpm += (new_bpm - self.bpm) * 0.3; // otherwise smooth the jitter
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Helpers (unchanged)
// ─────────────────────────────────────────────────────────────

fn make_window(kind: WindowFunction, n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    let denom = n.saturating_sub(1).max(1) as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    for (i, x) in w.iter_mut().enumerate() {
        let t = i as f32 / denom;
        *x = match kind {
            WindowFunction::Hann => 0.5 * (1.0 - (two_pi * t).cos()),
            WindowFunction::Hamming => 0.54 - 0.46 * (two_pi * t).cos(),
            WindowFunction::BlackmanHarris => {
                0.35875 - 0.48829 * (two_pi * t).cos()
                    + 0.14128 * (4.0 * std::f32::consts::PI * t).cos()
                    - 0.01168 * (6.0 * std::f32::consts::PI * t).cos()
            }
            WindowFunction::FlatTop => {
                0.21557895 - 0.41663158 * (two_pi * t).cos()
                    + 0.277263158 * (4.0 * std::f32::consts::PI * t).cos()
                    - 0.083578947 * (6.0 * std::f32::consts::PI * t).cos()
                    + 0.006947368 * (8.0 * std::f32::consts::PI * t).cos()
            }
            WindowFunction::Rectangular => 1.0,
        };
    }
    w
}

fn make_bin_ranges(
    fft_size: usize,
    num_bars: usize,
    scaling: FrequencyScaling,
    sample_rate: f32,
    min_freq: f32,
    max_freq: f32,
) -> Vec<(usize, usize)> {
    let nyquist = fft_size / 2;
    let bin_width = sample_rate / fft_size as f32;
    let min_bin = ((min_freq / bin_width).floor() as usize).max(1);
    let max_bin = ((max_freq / bin_width).ceil() as usize).min(nyquist);

    let mut ranges = Vec::with_capacity(num_bars);
    match scaling {
        FrequencyScaling::Linear => {
            for i in 0..num_bars {
                let s = min_bin + (i * (max_bin - min_bin)) / num_bars;
                let e = min_bin + ((i + 1) * (max_bin - min_bin)) / num_bars;
                ranges.push((s, e.max(s + 1).min(nyquist)));
            }
        }
        FrequencyScaling::Logarithmic => {
            let min_b = min_bin as f32;
            let max_b = max_bin as f32;
            for i in 0..num_bars {
                let lo = (min_b * (max_b / min_b).powf(i as f32 / num_bars as f32)).floor() as usize;
                let hi = (min_b * (max_b / min_b).powf((i + 1) as f32 / num_bars as f32)).ceil() as usize;
                let s = lo.max(min_bin);
                let e = hi.max(s + 1).min(max_bin);
                ranges.push((s, e));
            }
        }
    }
    ranges
}

fn centers_from_ranges(ranges: &[(usize, usize)], fft_size: usize, sample_rate: f32) -> Vec<f32> {
    let bin_width = sample_rate / fft_size as f32;
    ranges.iter().map(|(s, e)| {
        let center = (*s + *e) as f32 * 0.5;
        (center * bin_width).max(20.0)
    }).collect()
}

fn a_weighting_db(f: f32) -> f32 {
    let f = f.max(10.0);
    let f2 = f * f;
    let num = 12194.0_f32.powi(2) * f2.powi(2);
    let den = (f2 + 20.6_f32.powi(2))
        * ((f2 + 107.7_f32.powi(2)) * (f2 + 737.9_f32.powi(2))).sqrt()
        * (f2 + 12194.0_f32.powi(2));
    let ra = num / den;
    20.0 * ra.log10() + 2.0
}

fn k_weighting_db(f: f32) -> f32 {
    let f = f.max(10.0);
    let shelf = 4.0 * (1.0 - 1.0 / (1.0 + (f / 1500.0).powi(4)));
    let hp = -0.5 * (f / 60.0).powi(2) / (1.0 + (f / 60.0).powi(2));
    shelf + hp
}
