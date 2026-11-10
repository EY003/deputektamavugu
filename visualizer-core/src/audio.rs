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
    /// `on_frame` is invoked from the analysis thread whenever a new frame was
    /// published. The UI uses it to wake itself up from idle (no polling).
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
// Capture thread — fully event driven (no polling, no 1 ms sleeps)
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
        // Sleep inside the kernel until WASAPI signals new data. In loopback
        // mode nothing is signalled while no application is playing, so this
        // thread costs ~0 % when idle (it only wakes 4x/s on the timeout).
        let waited_at = Instant::now();
        if h_event.wait_for_event(250).is_err() && waited_at.elapsed() < Duration::from_millis(20) {
            // The wait failed instantly (device lost?) — never busy-loop.
            thread::sleep(Duration::from_millis(100));
        }

        // Drain every packet that is currently available.
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
            if sample_tx.send(mono.clone()).is_err() { return; }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Analysis thread
// ─────────────────────────────────────────────────────────────

/// Peak absolute sample value above which a chunk counts as "real audio".
const LOUD_THRESHOLD: f32 = 1.0e-4;
/// How long the input must be quiet (and the bars decayed) before we go idle.
const IDLE_AFTER: Duration = Duration::from_millis(700);

fn peak_abs(chunk: &[f32]) -> f32 {
    chunk.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

struct DcBlocker { alpha: f32, x: f32, y: f32 }
impl DcBlocker {
    fn new(sample_rate: f32) -> Self {
        // ~10 Hz cutoff at the current sample rate.
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
    let mut history: Vec<f32> = Vec::with_capacity(16_384);
    let mut dc = DcBlocker::new(sample_rate);
    // Spin only for the last 0.5 ms of each period instead of burning a core.
    let sleeper = spin_sleep::SpinSleeper::new(500_000);

    let mut last_tick = Instant::now();
    let mut last_loud = Instant::now();
    let mut last_chunk = Instant::now();
    let mut idle = false;
    let mut last_sent_zero = false;

    loop {
        let current = settings.load_full();

        // ── Deep idle: block until REAL audio arrives. 0 % CPU. ────────
        if idle {
            let mut chunk = match sample_rx.recv() {
                Ok(c) => c,
                Err(_) => return,
            };
            if peak_abs(&chunk) <= LOUD_THRESHOLD {
                continue; // digital silence is still streaming; stay asleep
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

        // ── Ingest whatever the capture thread produced ────────────────
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

        // WASAPI loopback delivers NO packets while nothing is playing. Feed
        // zeros so the bars decay smoothly instead of freezing on stale data.
        if !got_chunk && last_chunk.elapsed() > Duration::from_millis(60) {
            let n = ((dt_raw * sample_rate) as usize).min(16_384);
            history.resize(history.len() + n, 0.0);
        }

        let cap = (current.fft_size * 4).max(8192);
        if history.len() > cap {
            let drop_n = history.len() - cap;
            history.drain(0..drop_n);
        }

        // ── Analyse + publish ──────────────────────────────────────────
        if let Some(frame) = processor.process(&history, &current, dt) {
            let is_zero = frame.values.iter().all(|&v| v == 0.0)
                && frame.peaks.iter().all(|&v| v == 0.0);
            // Don't wake the UI to redraw an identical empty frame.
            if !(is_zero && last_sent_zero) {
                if frame_tx.try_send(frame).is_ok() {
                    last_sent_zero = is_zero;
                    on_frame();
                }
            }
        }

        // ── Enter deep idle once silent and fully decayed ──────────────
        if last_loud.elapsed() > IDLE_AFTER && processor.is_silent() {
            processor.snap_to_silence();
            let frame = processor.silent_frame();
            // Make sure the UI sees the final all-zero frame (it must not be
            // dropped, or stale bars would stay on screen forever).
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
// The processor
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
    bin_centers_hz: Vec<f32>,  // precomputed centre frequency for each bar

    bar_values: Vec<f32>,
    bar_peaks: Vec<f32>,
    bar_velocities: Vec<f32>,
    peak_hold_remaining: Vec<f32>,

    frame_history: VecDeque<Vec<f32>>,
    agc_envelope: f32,

    beat_energy_history: VecDeque<f32>,
    beat_times: VecDeque<f64>,
    bpm: f32,
    phase: f32,
    beat_envelope: f32,
    time_elapsed: f64,
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

        Self {
            sample_rate,
            cached_fft_size: fft_size,
            fft,
            window,
            cached_window_fn: window_fn,
            input_buffer,
            output_buffer,
            scratch_buffer,
            cached_num_bars: 80,
            cached_scaling: FrequencyScaling::Logarithmic,
            cached_min_freq: 30.0,
            cached_max_freq: 16000.0,
            bin_ranges,
            bin_centers_hz,
            bar_values: vec![0.0; 80],
            bar_peaks: vec![0.0; 80],
            bar_velocities: vec![0.0; 80],
            peak_hold_remaining: vec![0.0; 80],
            frame_history: VecDeque::with_capacity(16),
            agc_envelope: 1.0,
            beat_energy_history: VecDeque::with_capacity(240),
            beat_times: VecDeque::with_capacity(8),
            bpm: 0.0,
            phase: 0.0,
            beat_envelope: 0.0,
            time_elapsed: 0.0,
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

    /// True when every bar/peak has decayed to (almost) nothing.
    fn is_silent(&self) -> bool {
        self.bar_values.iter().all(|&v| v < 1.0e-3)
            && self.bar_peaks.iter().all(|&v| v < 1.0e-3)
            && self.beat_envelope < 1.0e-3
    }

    /// Hard-reset all decaying state to exact zero.
    fn snap_to_silence(&mut self) {
        for v in self.bar_values.iter_mut() { *v = 0.0; }
        for v in self.bar_peaks.iter_mut() { *v = 0.0; }
        for v in self.bar_velocities.iter_mut() { *v = 0.0; }
        for v in self.peak_hold_remaining.iter_mut() { *v = 0.0; }
        self.frame_history.clear();
        self.beat_energy_history.clear();
        self.beat_envelope = 0.0;
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
        for f in self.frame_history.iter_mut() { f.resize(n, 0.0); }
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

        // ── raw per-bar magnitudes (linear) ───────────────────────
        let mut raw_bars = vec![0.0f32; num_bars];
        for (i, range) in self.bin_ranges.iter().enumerate().take(num_bars) {
            let (start, end) = *range;
            let end = end.min(nyquist).max(start + 1);
            let mut mag = 0.0f32;
            for bin in start..end {
                mag += self.output_buffer[bin].norm();
            }
            mag /= (end - start) as f32;
            raw_bars[i] = (mag * norm + 1e-9).max(1e-9);
        }

        // ── frequency-domain smoothing (linear space) ─────────────
        if s.frequency_smoothing > 0.0 {
            let alpha = s.frequency_smoothing.clamp(0.0, 1.0);
            let mut tmp = raw_bars.clone();
            for i in 0..num_bars {
                let lo = i.saturating_sub(1);
                let hi = (i + 1).min(num_bars - 1);
                let avg = (raw_bars[lo] + raw_bars[i] * 2.0 + raw_bars[hi]) * 0.25;
                tmp[i] = raw_bars[i] * (1.0 - alpha) + avg * alpha;
            }
            raw_bars = tmp;
        }

        // ── harmonic emphasis (peak smearing) ─────────────────────
        if s.harmonic_emphasis && num_bars >= 5 {
            let mut bloom = raw_bars.clone();
            for i in 2..num_bars - 2 {
                let c = raw_bars[i];
                let l1 = raw_bars[i - 1];
                let r1 = raw_bars[i + 1];
                let l2 = raw_bars[i - 2];
                let r2 = raw_bars[i + 2];
                // Only bloom true local maxima above a floor
                if c > l1 && c > r1 && c > l2 && c > r2 && c > 0.005 {
                    let spread = c * 0.15;
                    bloom[i - 1] = bloom[i - 1].max(spread * 0.7);
                    bloom[i + 1] = bloom[i + 1].max(spread * 0.7);
                    bloom[i - 2] = bloom[i - 2].max(spread * 0.3);
                    bloom[i + 2] = bloom[i + 2].max(spread * 0.3);
                }
            }
            raw_bars = bloom;
        }

        // ── dB mapping with tilt / EQ / weighting ─────────────────
        let sens_norm = (s.sensitivity / 100.0).clamp(0.0, 1.0);
        let min_db_effective = s.min_decibels - (sens_norm - 0.5) * 20.0;
        let db_span = (s.max_decibels - min_db_effective).max(1.0);
        let target_ratio = db_span / s.target_dynamic_range_db.max(1.0);
        let gamma = s.response_gamma.max(0.1) * target_ratio;

        let mut mapped = vec![0.0f32; num_bars];
        for i in 0..num_bars {
            let db = 20.0 * raw_bars[i].log10();
            let center_freq = self.bin_centers_hz.get(i).copied().unwrap_or(1000.0);

            let octaves_from_1k = (center_freq / 1000.0).log2();
            let tilt_db = s.spectrum_tilt * octaves_from_1k;

            let eq_db = if center_freq < 250.0 {
                s.eq_bass_db
            } else if center_freq < 4000.0 {
                s.eq_mid_db
            } else {
                s.eq_treble_db
            };

            let weight_db = match s.weighting {
                WeightingCurve::None => 0.0,
                WeightingCurve::AWeighting => a_weighting_db(center_freq),
                WeightingCurve::KWeighting => k_weighting_db(center_freq),
            };

            let tilted_db = db + tilt_db + eq_db + weight_db;
            let mut n01 = ((tilted_db - min_db_effective) / db_span).clamp(0.0, 1.0);

            if n01 < s.noise_gate {
                n01 = 0.0;
            }
            mapped[i] = n01.powf(gamma);
        }

        // ── time-domain smoothing (average_size) ──────────────────
        let avg = s.average_size.max(1);
        while self.frame_history.len() >= avg { self.frame_history.pop_front(); }
        self.frame_history.push_back(mapped);

        let mut smoothed: Vec<f32> = {
            let n = num_bars;
            let count = self.frame_history.len() as f32;
            let mut out = vec![0.0f32; n];
            for frame in &self.frame_history {
                for i in 0..n.min(frame.len()) {
                    out[i] += frame[i];
                }
            }
            for x in &mut out { *x /= count.max(1.0); }
            out
        };

        // ── per-frame peak normalization ──────────────────────────
        if s.per_frame_normalization {
            let peak = smoothed.iter().cloned().fold(0.0f32, f32::max);
            if peak > 0.001 {
                for v in smoothed.iter_mut() { *v /= peak; }
            }
        }

        self.update_ballistics(&smoothed, s, dt);
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
            // Per-bar ballistics selection
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

        let mut bass_energy = 0.0;
        let mut mid_energy = 0.0;
        let mut treble_energy = 0.0;
        for bin in 1..nyquist {
            let f = bin as f32 * bin_width;
            let mag = self.output_buffer[bin].norm();
            if f < 250.0 { bass_energy += mag; }
            else if f < 4000.0 { mid_energy += mag; }
            else { treble_energy += mag; }
        }
        let energy = match s.beat_focus {
            BeatFocusMode::Bass => bass_energy,
            BeatFocusMode::Mid => mid_energy,
            BeatFocusMode::Treble => treble_energy,
        };

        let hist_len = (s.poll_rate_hz * 1.5).max(30.0) as usize;
        self.beat_energy_history.push_back(energy);
        while self.beat_energy_history.len() > hist_len {
            self.beat_energy_history.pop_front();
        }
        let local_avg = if self.beat_energy_history.is_empty() {
            energy
        } else {
            self.beat_energy_history.iter().sum::<f32>() / self.beat_energy_history.len() as f32
        };

        let sensitivity = s.beat_detection_sensitivity.clamp(0.0, 1.0);
        let threshold = local_avg * s.energy_threshold_multiplier.max(1.0) * (1.0 - sensitivity * 0.5);
        let is_beat = energy > threshold
            && energy > local_avg * 1.05
            && self.time_elapsed - self.beat_times.back().copied().unwrap_or(0.0) > 0.15;

        if is_beat {
            self.beat_times.push_back(self.time_elapsed);
            while self.beat_times.len() > 8 { self.beat_times.pop_front(); }
            self.recompute_bpm();
        }

        // Beat envelope (fast attack to 1.0, exponential release)
        if is_beat {
            self.beat_envelope = 1.0;
        } else if self.beat_envelope > 0.0 {
            let tau = (s.beat_envelope_decay_ms / 1000.0).max(0.001);
            self.beat_envelope *= (-dt / tau).exp();
            if self.beat_envelope < 0.001 { self.beat_envelope = 0.0; }
        }

        let period_s = if self.bpm > 20.0 { 60.0 / self.bpm } else { 0.5 };
        self.phase += dt / period_s;
        if self.phase >= 1.0 { self.phase -= 1.0; }

        if is_beat && s.bpm_lock_enabled {
            let gain = s.bpm_phase_smoothing.clamp(0.0, 1.0);
            let error = if self.phase < 0.5 { -self.phase } else { 1.0 - self.phase };
            self.phase = (self.phase + error * gain).rem_euclid(1.0);
        }

        (is_beat, self.phase)
    }

    fn recompute_bpm(&mut self) {
        if self.beat_times.len() < 2 { self.bpm = 0.0; return; }
        let mut intervals: Vec<f64> = self.beat_times.iter()
            .zip(self.beat_times.iter().skip(1))
            .map(|(a, b)| b - a).collect();
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = intervals[intervals.len() / 2];
        if median > 0.001 { self.bpm = (60.0 / median) as f32; }
    }
}

// ─────────────────────────────────────────────────────────────
// Helpers
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

/// Standard A-weighting in dB (IEC 61672:2003).
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

/// Simplified K-weighting (ITU-R BS.1770) — high-shelf + gentle high-pass.
fn k_weighting_db(f: f32) -> f32 {
    let f = f.max(10.0);
    let shelf = 4.0 * (1.0 - 1.0 / (1.0 + (f / 1500.0).powi(4)));
    let hp = -0.5 * (f / 60.0).powi(2) / (1.0 + (f / 60.0).powi(2));
    shelf + hp
}
