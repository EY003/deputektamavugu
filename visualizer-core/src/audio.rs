use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam_channel::{
    Receiver, RecvTimeoutError, SendTimeoutError, Sender, TrySendError,
};
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use wasapi::*;

use crate::config::{
    AudioSettings, BeatFocusMode, ChannelMixMode, DecayMode, FrequencyScaling,
    WeightingCurve, WindowFunction,
};

const MIN_BEAT_GAP_S: f64 = 0.25;
const BPM_FOLD_MIN: f64 = 70.0;
const BPM_FOLD_MAX: f64 = 180.0;
const BEAT_MEMORY_S: f64 = 8.0;
const BEAT_TIMEOUT_S: f64 = 5.0;
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
    stop: Arc<AtomicBool>,
}

impl AudioEngine {
    pub fn new<F>(initial: AudioSettings, on_frame: F) -> Self
    where
        F: Fn() + Send + 'static,
    {
        let settings = Arc::new(ArcSwap::from_pointee(initial));
        let stop = Arc::new(AtomicBool::new(false));

        let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
        let (sample_tx, sample_rx) = crossbeam_channel::bounded::<Vec<f32>>(32);
        let (ctrl_tx, ctrl_rx) = crossbeam_channel::unbounded::<ControlMsg>();
        let (info_tx, info_rx) = crossbeam_channel::bounded(1);

        let sample_rate_shared = Arc::new(AtomicU32::new(48_000f32.to_bits()));

        let settings_for_capture = settings.clone();
        let stop_cap = stop.clone();
        let sr_cap = sample_rate_shared.clone();
        thread::spawn(move || {
            capture_loop(settings_for_capture, sample_tx, ctrl_tx, info_tx, sr_cap, stop_cap);
        });

        let (device_name, sample_rate) = info_rx
            .recv()
            .unwrap_or_else(|_| ("Unknown".to_string(), 48000.0));
        sample_rate_shared.store(sample_rate.to_bits(), Ordering::Relaxed);

        let settings_for_analysis = settings.clone();
        let stop_an = stop.clone();
        let sr_an = sample_rate_shared.clone();
        thread::spawn(move || {
            analysis_loop(
                settings_for_analysis, sample_rx, ctrl_rx, frame_tx, sr_an, stop_an, on_frame,
            );
        });

        Self { device_name, frame_rx, settings, stop }
    }

    /// Signal the background capture / analysis threads to stop.
    ///
    /// Threads exit on their own when the process terminates, so you only
    /// need this if you want to tear down an engine while the app is still
    /// running (e.g. re-creating it for device switching). Once called, this
    /// engine's `frame_rx` stops producing frames.
    pub fn shutdown(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

// ─────────────────────────────────────────────────────────────
// Capture
// ─────────────────────────────────────────────────────────────

#[derive(Debug)]
enum ControlMsg {
    ResetDcBlocker,
}

#[derive(Debug)]
enum SessionEnd {
    Stopped,
    Disconnected,
    Invalidated,
}

fn capture_loop(
    settings: Arc<ArcSwap<AudioSettings>>,
    sample_tx: Sender<Vec<f32>>,
    ctrl_tx: Sender<ControlMsg>,
    info_tx: Sender<(String, f32)>,
    sample_rate_shared: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
) {
    let _ = initialize_mta();
    let mut sent_info = false;
    let mut backoff = Duration::from_millis(500);

    loop {
        if stop.load(Ordering::Relaxed) { return; }
        match capture_session(
            &settings, &sample_tx, &ctrl_tx, &info_tx,
            &sample_rate_shared, &stop, &mut sent_info,
        ) {
            SessionEnd::Stopped | SessionEnd::Disconnected => return,
            SessionEnd::Invalidated => {
                eprintln!("WASAPI: device invalidated, retrying in {:?}", backoff);
                thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(5));
            }
        }
    }
}

fn capture_session(
    settings: &Arc<ArcSwap<AudioSettings>>,
    sample_tx: &Sender<Vec<f32>>,
    ctrl_tx: &Sender<ControlMsg>,
    info_tx: &Sender<(String, f32)>,
    sample_rate_shared: &Arc<AtomicU32>,
    stop: &Arc<AtomicBool>,
    sent_info: &mut bool,
) -> SessionEnd {
    let device = match get_default_device(&Direction::Render) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("WASAPI: get_default_device failed: {e:?}");
            if !*sent_info { let _ = info_tx.send(("Unknown".into(), 48_000.0)); }
            return SessionEnd::Invalidated;
        }
    };
    let mut audio_client = match device.get_iaudioclient() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("WASAPI: get_iaudioclient failed: {e:?}");
            if !*sent_info { let _ = info_tx.send(("Unknown".into(), 48_000.0)); }
            return SessionEnd::Invalidated;
        }
    };

    let device_name = device.get_friendlyname().unwrap_or_else(|_| "Default Device".into());
    let format = match audio_client.get_mixformat() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WASAPI: get_mixformat failed: {e:?}");
            if !*sent_info { let _ = info_tx.send((device_name, 48_000.0)); }
            return SessionEnd::Invalidated;
        }
    };
    let sample_rate = format.get_samplespersec() as f32;
    let channels = format.get_nchannels() as usize;
    let blockalign = format.get_blockalign() as usize;
    let bits = format.get_bitspersample() as usize;

    if bits != 32 {
        eprintln!(
            "WASAPI: non-float mix format ({bits} bits). \
             Samples-per-frame assumes 32-bit float; audio will be garbled."
        );
    }
    let samples_per_frame = (blockalign / 4).max(1).max(channels);

    let (_, min_time) = match audio_client.get_periods() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("WASAPI: get_periods failed: {e:?}");
            (100_000i64, 100_000i64)
        }
    };
    if let Err(e) = audio_client.initialize_client(
        &format, min_time, &Direction::Capture, &ShareMode::Shared, true,
    ) {
        eprintln!("WASAPI: initialize_client failed: {e:?}");
        if !*sent_info { let _ = info_tx.send((device_name, sample_rate)); }
        return SessionEnd::Invalidated;
    }

    let h_event = match audio_client.set_get_eventhandle() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("WASAPI: set_get_eventhandle failed: {e:?}");
            if !*sent_info { let _ = info_tx.send((device_name, sample_rate)); }
            return SessionEnd::Invalidated;
        }
    };
    let capture_client = match audio_client.get_audiocaptureclient() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("WASAPI: get_audiocaptureclient failed: {e:?}");
            if !*sent_info { let _ = info_tx.send((device_name, sample_rate)); }
            return SessionEnd::Invalidated;
        }
    };
    if let Err(e) = audio_client.start_stream() {
        eprintln!("WASAPI: start_stream failed: {e:?}");
        if !*sent_info { let _ = info_tx.send((device_name, sample_rate)); }
        return SessionEnd::Invalidated;
    }

    if !*sent_info {
        let _ = info_tx.send((device_name, sample_rate));
        *sent_info = true;
    }
    sample_rate_shared.store(sample_rate.to_bits(), Ordering::Relaxed);

    let mut sample_buffer: Vec<f32> = Vec::new();
    let mut mono: Vec<f32> = Vec::with_capacity(4096);

    loop {
        if stop.load(Ordering::Relaxed) {
            let _ = audio_client.stop_stream();
            return SessionEnd::Stopped;
        }

        let waited_at = Instant::now();
        if h_event.wait_for_event(250).is_err() && waited_at.elapsed() < Duration::from_millis(20) {
            thread::sleep(Duration::from_millis(100));
        }

        loop {
            let packet_size = match capture_client.get_next_nbr_frames() {
                Ok(Some(n)) => n,
                Ok(None) => 0,
                Err(e) => {
                    eprintln!("WASAPI: get_next_nbr_frames error: {e:?}");
                    let _ = audio_client.stop_stream();
                    return SessionEnd::Invalidated;
                }
            };
            if packet_size == 0 { break; }

            let needed = packet_size as usize * samples_per_frame;
            sample_buffer.resize(needed, 0.0);
            let byte_slice: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    sample_buffer.as_mut_ptr() as *mut u8,
                    needed * std::mem::size_of::<f32>(),
                )
            };
            let (frames_read, flags) = match capture_client.read_from_device(byte_slice) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("WASAPI: read_from_device error: {e:?}");
                    let _ = audio_client.stop_stream();
                    return SessionEnd::Invalidated;
                }
            };
            if frames_read == 0 { break; }

            // BufferFlags in wasapi is a plain struct with `bool` fields.
            if flags.data_discontinuity {
                let _ = ctrl_tx.try_send(ControlMsg::ResetDcBlocker);
            }
            if flags.silent {
                sample_buffer[..needed].fill(0.0);
            }

            let valid_samples = (frames_read as usize * samples_per_frame).min(sample_buffer.len());
            let samples = &sample_buffer[..valid_samples];

            let mix_mode = settings.load().channel_mix;
            mono.clear();
            mono.reserve(valid_samples / channels.max(1));
            for chunk in samples.chunks_exact(channels.max(1)) {
                mono.push(downmix(chunk, mix_mode));
            }
            if mono.is_empty() { continue; }
            let out = std::mem::replace(&mut mono, Vec::with_capacity(4096));
            match sample_tx.send_timeout(out, Duration::from_millis(50)) {
                Ok(()) => {}
                Err(SendTimeoutError::Timeout(_)) => { /* analysis is slow; drop */ }
                Err(SendTimeoutError::Disconnected(_)) => {
                    let _ = audio_client.stop_stream();
                    return SessionEnd::Disconnected;
                }
            }
        }
    }
}

fn downmix(chunk: &[f32], mode: ChannelMixMode) -> f32 {
    let n = chunk.len();
    match n {
        0 => 0.0,
        1 => chunk[0],
        _ => match mode {
            ChannelMixMode::Left => chunk[0],
            ChannelMixMode::Right => chunk[1],
            ChannelMixMode::Side => (chunk[0] - chunk[1]) * 0.5,
            ChannelMixMode::Mono | ChannelMixMode::Mid => {
                if n == 2 {
                    (chunk[0] + chunk[1]) * 0.5
                } else {
                    let mut sum = chunk[0] + chunk[1];
                    let mut wsum = 2.0;
                    if n >= 3 { sum += chunk[2] * 0.707; wsum += 0.707; }
                    if n >= 4 { sum += chunk[3] * 0.5;   wsum += 0.5; }
                    if n >= 6 { sum += (chunk[4] + chunk[5]) * 0.707; wsum += 1.414; }
                    for i in 6..n { sum += chunk[i] * 0.5; wsum += 0.5; }
                    sum / wsum
                }
            }
        },
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
    fn reset(&mut self) { self.x = 0.0; self.y = 0.0; }
}

fn analysis_loop<F: Fn() + Send + 'static>(
    settings: Arc<ArcSwap<AudioSettings>>,
    sample_rx: Receiver<Vec<f32>>,
    ctrl_rx: Receiver<ControlMsg>,
    frame_tx: Sender<BarFrame>,
    sample_rate_shared: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
    on_frame: F,
) {
    let mut sample_rate = f32::from_bits(sample_rate_shared.load(Ordering::Relaxed));
    let mut processor = AudioProcessor::new(sample_rate);
    let mut history: Vec<f32> = Vec::with_capacity(16_384);
    let mut dc = DcBlocker::new(sample_rate);
    let sleeper = spin_sleep::SpinSleeper::new(500_000);

    let mut last_tick = Instant::now();
    let mut last_loud = Instant::now();
    let mut last_chunk = Instant::now();
    let mut idle = false;
    let mut last_sent_zero = false;
    let mut pending_beat = false;
    let mut silent_frame_pending: Option<BarFrame> = None;

    loop {
        if stop.load(Ordering::Relaxed) { return; }

        while let Ok(msg) = ctrl_rx.try_recv() {
            match msg {
                ControlMsg::ResetDcBlocker => dc.reset(),
            }
        }

        let new_sr = f32::from_bits(sample_rate_shared.load(Ordering::Relaxed));
        if (new_sr - sample_rate).abs() > 0.5 {
            eprintln!("Audio: sample rate changed {sample_rate} -> {new_sr}, rebuilding processor");
            sample_rate = new_sr;
            processor = AudioProcessor::new(sample_rate);
            dc = DcBlocker::new(sample_rate);
            history.clear();
        }

        let current = settings.load_full();

        if let Some(frame) = silent_frame_pending.take() {
            match frame_tx.send_timeout(frame.clone(), Duration::from_millis(100)) {
                Ok(()) => { last_sent_zero = true; on_frame(); }
                Err(SendTimeoutError::Timeout(_)) => { silent_frame_pending = Some(frame); }
                Err(SendTimeoutError::Disconnected(_)) => return,
            }
        }

        if idle {
            if current.num_bars != processor.bar_values.len()
                && silent_frame_pending.is_none()
            {
                processor.update_initial_state(&current);
                let frame = processor.silent_frame();
                match frame_tx.try_send(frame.clone()) {
                    Ok(()) => { last_sent_zero = true; on_frame(); }
                    Err(TrySendError::Full(_)) => { silent_frame_pending = Some(frame); }
                    Err(TrySendError::Disconnected(_)) => return,
                }
            }

            let mut chunk = match sample_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(c) => c,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return,
            };
            if peak_abs(&chunk) <= LOUD_THRESHOLD { continue; }
            idle = false;
            let now = Instant::now();
            last_loud = now;
            last_chunk = now;
            last_tick = now;
            if current.dc_block { dc.apply(&mut chunk); }
            history.clear();
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
            history.resize(history.len() + n, 0.0);
        }

        let cap = (current.fft_size * 4).max(8192);
        if history.len() > cap {
            let drop_n = history.len() - cap;
            history.drain(0..drop_n);
        }

        if let Some(mut frame) = processor.process(&history, &current, dt) {
            if pending_beat { frame.beat = true; }
            if frame.beat { pending_beat = true; }

            let is_zero = frame.values.iter().all(|&v| v == 0.0)
                && frame.peaks.iter().all(|&v| v == 0.0);
            let should_send = !(is_zero && last_sent_zero) || frame.beat;

            if should_send {
                match frame_tx.try_send(frame) {
                    Ok(()) => {
                        last_sent_zero = is_zero;
                        pending_beat = false;
                        on_frame();
                    }
                    Err(TrySendError::Full(_)) => { /* keep pending_beat */ }
                    Err(TrySendError::Disconnected(_)) => return,
                }
            }
        }

        if last_loud.elapsed() > IDLE_AFTER && processor.is_silent() {
            processor.snap_to_silence();
            let frame = processor.silent_frame();
            match frame_tx.try_send(frame.clone()) {
                Ok(()) => { last_sent_zero = true; on_frame(); }
                Err(TrySendError::Full(_)) => {
                    silent_frame_pending = Some(frame);
                    on_frame();
                }
                Err(TrySendError::Disconnected(_)) => return,
            }
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
// Processor
// ─────────────────────────────────────────────────────────────

struct AudioProcessor {
    sample_rate: f32,
    cached_fft_size: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    window_gain: f32,
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

    raw_bars: Vec<f32>,
    tmp_bars: Vec<f32>,
    bloom_bars: Vec<f32>,
    mapped: Vec<f32>,
    smoothed: Vec<f32>,

    frame_ring: Vec<f32>,
    frame_ring_pos: usize,
    frame_ring_count: usize,
    cached_avg: usize,

    agc_envelope: f32,

    beat_energy_history: VecDeque<f32>,
    beat_times: VecDeque<f64>,
    bpm: f32,
    phase: f32,
    beat_envelope: f32,
    time_elapsed: f64,

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
        let window_gain = compute_window_gain(&window);
        let num_bars = 80;
        let bin_ranges = make_bin_ranges(
            fft_size, num_bars, FrequencyScaling::Logarithmic, sample_rate, 30.0, 16_000.0,
        );
        let bin_centers_hz = compute_bar_centers_hz(
            num_bars, FrequencyScaling::Logarithmic, 30.0, 16_000.0,
        );

        Self {
            sample_rate,
            cached_fft_size: fft_size,
            fft, window, window_gain,
            cached_window_fn: window_fn,
            input_buffer, output_buffer, scratch_buffer,
            cached_num_bars: num_bars,
            cached_scaling: FrequencyScaling::Logarithmic,
            cached_min_freq: 30.0,
            cached_max_freq: 16_000.0,
            bin_ranges, bin_centers_hz,
            bar_values: vec![0.0; num_bars],
            bar_peaks: vec![0.0; num_bars],
            bar_velocities: vec![0.0; num_bars],
            peak_hold_remaining: vec![0.0; num_bars],
            raw_bars: vec![0.0; num_bars],
            tmp_bars: vec![0.0; num_bars],
            bloom_bars: vec![0.0; num_bars],
            mapped: vec![0.0; num_bars],
            smoothed: vec![0.0; num_bars],
            frame_ring: vec![0.0; num_bars],
            frame_ring_pos: 0,
            frame_ring_count: 0,
            cached_avg: 1,
            agc_envelope: 1.0,
            beat_energy_history: VecDeque::with_capacity(240),
            beat_times: VecDeque::with_capacity(8),
            bpm: 0.0,
            phase: 0.0,
            beat_envelope: 0.0,
            time_elapsed: 0.0,
            beat_armed: true,
            bar_offset_db: Vec::new(),
            offset_centers: Vec::new(),
            offset_params: [f32::NAN; 4],
            offset_weighting: u8::MAX,
        }
    }

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
            let bm = smoothstep(f, 125.0, 500.0);
            let mt = smoothstep(f, 2000.0, 8000.0);
            let eq_db = s.eq_bass_db * (1.0 - bm)
                + s.eq_mid_db * (bm - mt)
                + s.eq_treble_db * mt;
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
            if !s.fft_size.is_power_of_two() || s.fft_size < 256 {
                return;
            }
            let mut planner = RealFftPlanner::new();
            self.fft = planner.plan_fft_forward(s.fft_size);
            self.input_buffer = self.fft.make_input_vec();
            self.output_buffer = self.fft.make_output_vec();
            self.scratch_buffer = self.fft.make_scratch_vec();
            self.cached_fft_size = s.fft_size;
            self.cached_window_fn = s.window_function;
            self.window = make_window(s.window_function, s.fft_size);
            self.window_gain = compute_window_gain(&self.window);
        } else if s.window_function != self.cached_window_fn {
            self.cached_window_fn = s.window_function;
            self.window = make_window(s.window_function, s.fft_size);
            self.window_gain = compute_window_gain(&self.window);
        }

        if s.num_bars != self.cached_num_bars
            || s.frequency_scaling != self.cached_scaling
            || s.min_frequency_hz != self.cached_min_freq
            || s.max_frequency_hz != self.cached_max_freq
        {
            if s.num_bars == 0 || s.min_frequency_hz >= s.max_frequency_hz {
                return;
            }
            self.cached_num_bars = s.num_bars;
            self.cached_scaling = s.frequency_scaling;
            self.cached_min_freq = s.min_frequency_hz;
            self.cached_max_freq = s.max_frequency_hz;
            self.bin_ranges = make_bin_ranges(
                s.fft_size, s.num_bars, s.frequency_scaling,
                self.sample_rate, s.min_frequency_hz, s.max_frequency_hz,
            );
            self.bin_centers_hz = compute_bar_centers_hz(
                s.num_bars, s.frequency_scaling, s.min_frequency_hz, s.max_frequency_hz,
            );
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
        self.beat_envelope = 0.0;
        self.beat_times.clear();
        self.beat_armed = true;
        self.bpm = 0.0;
        self.agc_envelope = 1.0;
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
        self.frame_ring.resize(n, 0.0);
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

        let raw_rms = {
            let sum_sq: f32 = window_slice.iter().map(|x| x * x).sum();
            (sum_sq / n_fft as f32).sqrt()
        };
        if s.agc_enabled && raw_rms > 1e-4 {
            let desired = (0.08 / raw_rms.max(1e-6)).clamp(0.01, 20.0);
            let coeff = if desired < self.agc_envelope {
                s.agc_attack.clamp(0.0001, 1.0)
            } else {
                s.agc_release.clamp(0.0001, 1.0)
            };
            let alpha = 1.0 - (1.0 - coeff).powf(dt * 60.0);
            self.agc_envelope += (desired - self.agc_envelope) * alpha;
            self.agc_envelope = self.agc_envelope.clamp(0.01, 20.0);
        } else if !s.agc_enabled {
            self.agc_envelope = 1.0;
        }
        let agc_gain = self.agc_envelope.clamp(0.01, 20.0);
        let user_gain = s.raw_gain.max(0.01) * agc_gain;

        self.input_buffer.copy_from_slice(window_slice);
        for i in 0..n_fft {
            self.input_buffer[i] *= self.window[i];
        }
        if self.fft.process_with_scratch(
            &mut self.input_buffer, &mut self.output_buffer, &mut self.scratch_buffer,
        ).is_err() {
            return None;
        }

        let num_bars = self.cached_num_bars;
        let nyquist = n_fft / 2;
        let norm = self.window_gain * user_gain;
        let bin_width = self.sample_rate / n_fft as f32;

        for i in 0..num_bars {
            let (start, end) = self.bin_ranges[i];
            let end = end.min(nyquist).max(start + 1);
            let mag;
            if end - start == 1 {
                let center_hz = self.bin_centers_hz.get(i).copied().unwrap_or(bin_width);
                let pos = center_hz / bin_width;
                let k = (pos.floor() as usize).min(nyquist);
                let t = (pos - k as f32).clamp(0.0, 1.0);
                let k1 = (k + 1).min(nyquist);
                mag = self.output_buffer[k].norm() * (1.0 - t)
                    + self.output_buffer[k1].norm() * t;
            } else {
                let mut sum = 0.0f32;
                for bin in start..end {
                    sum += self.output_buffer[bin].norm();
                }
                mag = sum / (end - start) as f32;
            }
            self.raw_bars[i] = (mag * norm + 1e-9).max(1e-9);
        }

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
            if n01 < s.noise_gate { n01 = 0.0; }
            self.mapped[i] = n01.powf(gamma);
        }

        let avg = ((s.average_size as f32 * s.poll_rate_hz / 60.0).round() as usize).max(1);
        let needed = avg * num_bars;
        if self.frame_ring.len() != needed || self.cached_avg != avg {
            self.frame_ring.resize(needed, 0.0);
            self.frame_ring_pos = 0;
            self.frame_ring_count = 0;
            self.cached_avg = avg;
        }
        if self.frame_ring_count < avg { self.frame_ring_count += 1; }
        let base = self.frame_ring_pos * num_bars;
        self.frame_ring[base..base + num_bars].copy_from_slice(&self.mapped);
        self.frame_ring_pos = (self.frame_ring_pos + 1) % avg;

        self.smoothed.fill(0.0);
        for f in 0..self.frame_ring_count {
            let offset = f * num_bars;
            for i in 0..num_bars {
                self.smoothed[i] += self.frame_ring[offset + i];
            }
        }
        let inv = 1.0 / self.frame_ring_count as f32;
        for v in self.smoothed.iter_mut() { *v *= inv; }

        if s.per_frame_normalization {
            let peak = self.smoothed.iter().cloned().fold(0.0f32, f32::max);
            if peak > 0.001 {
                for v in self.smoothed.iter_mut() { *v /= peak; }
            }
        }

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

            let attack_alpha = if atk_ms <= 0.0 { 1.0 }
                else { 1.0 - (-dt * 1000.0 / atk_ms).exp() };
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
                        let next = prev - self.bar_velocities[i] * dt;
                        if next <= input {
                            self.bar_velocities[i] = 0.0;
                            input
                        } else {
                            next
                        }
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

        let bass_end = ((250.0 / bin_width).ceil() as usize).clamp(1, nyquist);
        let mid_end = ((4000.0 / bin_width).ceil() as usize).clamp(bass_end, nyquist);
        let (lo, hi) = match s.beat_focus {
            BeatFocusMode::Bass => (1, bass_end),
            BeatFocusMode::Mid => (bass_end, mid_end),
            BeatFocusMode::Treble => (mid_end, nyquist),
        };
        let energy: f32 = self.output_buffer[lo..hi.max(lo)].iter().map(|c| c.norm()).sum();

        let hist_len = (s.poll_rate_hz * 1.5).max(30.0) as usize;
        while self.beat_energy_history.len() >= hist_len {
            self.beat_energy_history.pop_front();
        }
        self.beat_energy_history.push_back(energy);

        let min_hist = (s.poll_rate_hz * 0.5).max(15.0) as usize;
        if self.beat_energy_history.len() < min_hist {
            if self.beat_envelope > 0.0 {
                let tau = (s.beat_envelope_decay_ms / 1000.0).max(0.001);
                self.beat_envelope *= (-dt / tau).exp();
                if self.beat_envelope < 0.001 { self.beat_envelope = 0.0; }
            }
            let period_s = if self.bpm > 20.0 { 60.0 / self.bpm } else { 0.5 };
            self.phase = (self.phase + dt / period_s).rem_euclid(1.0);
            return (false, self.phase);
        }

        let local_avg = self.beat_energy_history.iter().sum::<f32>()
            / self.beat_energy_history.len() as f32;

        let sensitivity = s.beat_detection_sensitivity.clamp(0.0, 1.0);
        let threshold = local_avg * s.energy_threshold_multiplier.max(1.0)
            * (1.0 - sensitivity * 0.5);
        let above = energy > threshold && energy > local_avg * 1.05;

        if energy < threshold * 0.9 { self.beat_armed = true; }
        let since_last = self.time_elapsed
            - self.beat_times.back().copied().unwrap_or(f64::NEG_INFINITY);
        let is_beat = above && self.beat_armed && since_last > MIN_BEAT_GAP_S;
        if above { self.beat_armed = false; }

        if is_beat {
            self.beat_times.push_back(self.time_elapsed);
            while self.beat_times.len() > MAX_BEATS {
                self.beat_times.pop_front();
            }
            self.recompute_bpm();
        }

        while let Some(&t) = self.beat_times.front() {
            if self.time_elapsed - t > BEAT_MEMORY_S {
                self.beat_times.pop_front();
            } else { break; }
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
            if self.beat_envelope < 0.001 { self.beat_envelope = 0.0; }
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
        if self.beat_times.len() < 3 { return; }
        let min_iv = 60.0 / BPM_FOLD_MAX;
        let max_iv = 60.0 / BPM_FOLD_MIN;

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
        if iv.len() < 2 { return; }
        iv.sort_by(|a, b| a.total_cmp(b));
        let median = iv[iv.len() / 2];

        let (sum, n) = iv
            .iter()
            .filter(|d| (**d - median).abs() <= median * 0.12)
            .fold((0.0f64, 0usize), |(s, n), d| (s + d, n + 1));
        if n == 0 { return; }
        let new_bpm = (60.0 / (sum / n as f64)) as f32;

        if self.bpm <= 0.0 || (new_bpm - self.bpm).abs() / self.bpm > 0.12 {
            self.bpm = new_bpm;
        } else {
            self.bpm += (new_bpm - self.bpm) * 0.3;
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

fn compute_window_gain(window: &[f32]) -> f32 {
    let sum: f32 = window.iter().sum();
    if sum > 1e-9 { 2.0 / sum } else { 2.0 / window.len().max(1) as f32 }
}

fn make_window(kind: WindowFunction, n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    let n_f = n.max(1) as f32;
    let two_pi = 2.0 * std::f32::consts::PI;
    for (i, x) in w.iter_mut().enumerate() {
        let t = i as f32 / n_f;
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
    let nyquist_hz = bin_width * nyquist as f32;
    let min_freq = min_freq.clamp(bin_width, nyquist_hz - bin_width);
    let max_freq = max_freq.clamp(min_freq + bin_width, nyquist_hz);
    let min_bin = ((min_freq / bin_width).floor() as usize).max(1);
    let max_bin = ((max_freq / bin_width).ceil() as usize)
        .min(nyquist)
        .max(min_bin + 1);

    let mut ranges = Vec::with_capacity(num_bars);
    match scaling {
        FrequencyScaling::Linear => {
            let span = max_bin.saturating_sub(min_bin);
            for i in 0..num_bars {
                let s = min_bin + (i * span) / num_bars;
                let e = min_bin + ((i + 1) * span) / num_bars;
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

fn compute_bar_centers_hz(
    num_bars: usize,
    scaling: FrequencyScaling,
    min_freq: f32,
    max_freq: f32,
) -> Vec<f32> {
    if num_bars == 0 || max_freq <= min_freq { return Vec::new(); }
    let ratio = (max_freq / min_freq).powf(1.0 / num_bars as f32);
    (0..num_bars).map(|i| {
        match scaling {
            FrequencyScaling::Linear => {
                let step = (max_freq - min_freq) / num_bars as f32;
                (min_freq + (i as f32 + 0.5) * step).max(20.0)
            }
            FrequencyScaling::Logarithmic => {
                (min_freq * ratio.powf(i as f32 + 0.5)).max(20.0)
            }
        }
    }).collect()
}

fn smoothstep(x: f32, a: f32, b: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

fn biquad_db(f: f32, fs: f32, b: [f32; 3], a: [f32; 3]) -> f32 {
    let w = 2.0 * std::f32::consts::PI * f / fs;
    let (c1, s1) = (w.cos(), w.sin());
    let (c2, s2) = ((2.0 * w).cos(), (2.0 * w).sin());
    let nr = b[0] + b[1] * c1 + b[2] * c2;
    let ni = -(b[1] * s1 + b[2] * s2);
    let dr = a[0] + a[1] * c1 + a[2] * c2;
    let di = -(a[1] * s1 + a[2] * s2);
    let num = nr * nr + ni * ni;
    let den = dr * dr + di * di;
    if den < 1e-20 { return 0.0; }
    10.0 * (num / den).log10()
}

fn k_weighting_db(f: f32) -> f32 {
    const FS: f32 = 48_000.0;
    let shelf = biquad_db(
        f, FS,
        [1.535_124_86, -2.691_696_19, 1.198_392_81],
        [1.0, -1.690_659_29, 0.732_480_77],
    );
    let hp = biquad_db(
        f, FS,
        [1.0, -2.0, 1.0],
        [1.0, -1.990_047_45, 0.990_072_25],
    );
    shelf + hp
}
