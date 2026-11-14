use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};
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
    AudioSettings, BpmMultiplier, ChannelMixMode, DecayMode,
    FrequencyScaling, WeightingCurve, WindowFunction,
};

const MIN_BEAT_GAP_S: f64 = 0.25;
const BPM_FOLD_MIN: f64 = 70.0;
const BPM_FOLD_MAX: f64 = 180.0;
const BEAT_MEMORY_S: f64 = 8.0;
const BEAT_TIMEOUT_S: f64 = 5.0;
const MAX_BEATS: usize = 12;
const WAVEFORM_LEN: usize = 1024;
const SPECTRUM_ROW_LEN: usize = 128;
const DEVICE_POLL_INTERVAL: Duration = Duration::from_secs(2);

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Onsets {
    pub kick: bool,
    pub snare: bool,
    pub hat: bool,
    pub flux: f32,
}

#[derive(Clone, Debug)]
pub struct BarFrame {
    pub values: Vec<f32>,
    pub peaks: Vec<f32>,
    pub beat: bool,
    pub beat_predicted: bool,
    pub beat_envelope: f32,
    pub beat_in: f32,
    pub bpm: f32,
    pub phase: f32,
    pub onsets: Onsets,
    pub rms: f32,
    pub peak: f32,
    pub band_levels: [f32; 4],
    pub chroma: [f32; 12],
    pub dominant_pitch_hz: f32,
    pub dominant_pitch_midi: f32,
    pub waveform: Vec<f32>,
    pub spectrum_row: Vec<f32>,
    pub capture_instant: Instant,
}

impl BarFrame {
    pub fn zeros(num_bars: usize) -> Self {
        Self {
            values: vec![0.0; num_bars],
            peaks: vec![0.0; num_bars],
            beat: false, beat_predicted: false, beat_envelope: 0.0, beat_in: 0.0,
            bpm: 0.0, phase: 0.0,
            onsets: Onsets { kick: false, snare: false, hat: false, flux: 0.0 },
            rms: 0.0, peak: 0.0, band_levels: [0.0; 4], chroma: [0.0; 12],
            dominant_pitch_hz: 0.0, dominant_pitch_midi: 0.0,
            waveform: vec![0.0; WAVEFORM_LEN],
            spectrum_row: vec![0.0; SPECTRUM_ROW_LEN],
            capture_instant: Instant::now(),
        }
    }

    /// Linear interpolate this frame toward `next` by `alpha ∈ [0,1]`.
    /// Arrays with mismatched lengths are handled gracefully (shorter wins).
    pub fn lerp_toward(&self, next: &BarFrame, alpha: f32) -> BarFrame {
        let a = alpha.clamp(0.0, 1.0);
        let blend = |x: f32, y: f32| x * (1.0 - a) + y * a;
        let blend_vec = |xs: &[f32], ys: &[f32]| -> Vec<f32> {
            let n = xs.len().min(ys.len());
            let mut out = Vec::with_capacity(n);
            for i in 0..n { out.push(blend(xs[i], ys[i])); }
            out
        };
        let mut chroma = [0.0f32; 12];
        for i in 0..12 { chroma[i] = blend(self.chroma[i], next.chroma[i]); }
        let mut bands = [0.0f32; 4];
        for i in 0..4 { bands[i] = blend(self.band_levels[i], next.band_levels[i]); }
        BarFrame {
            values: blend_vec(&self.values, &next.values),
            peaks: blend_vec(&self.peaks, &next.peaks),
            beat: next.beat, beat_predicted: next.beat_predicted,
            beat_envelope: blend(self.beat_envelope, next.beat_envelope),
            beat_in: next.beat_in,
            bpm: next.bpm, phase: next.phase,
            onsets: next.onsets.clone(),
            rms: blend(self.rms, next.rms),
            peak: blend(self.peak, next.peak),
            band_levels: bands,
            chroma,
            dominant_pitch_hz: next.dominant_pitch_hz,
            dominant_pitch_midi: next.dominant_pitch_midi,
            waveform: next.waveform.clone(),
            spectrum_row: blend_vec(&self.spectrum_row, &next.spectrum_row),
            capture_instant: next.capture_instant,
        }
    }
}

#[derive(Default)]
pub struct AudioStats {
    pub frames_captured: AtomicU64,
    pub frames_dropped: AtomicU64,
    pub analysis_time_us: AtomicU64,
    pub capture_time_us: AtomicU64,
    pub sample_rate_bits: AtomicU32,
    pub fft_size: AtomicU32,
    pub channels: AtomicU16,
    pub bits_per_sample: AtomicU16,
    pub onset_count: AtomicU64,
}

pub struct AudioEngine {
    pub device_name: String,
    pub frame_rx: Receiver<BarFrame>,
    pub settings: Arc<ArcSwap<AudioSettings>>,
    pub stats: Arc<AudioStats>,
    tap_tx: Sender<Instant>,
    stop: Arc<AtomicBool>,
}

impl AudioEngine {
    pub fn new<F>(initial: AudioSettings, on_frame: F) -> Self
    where F: Fn() + Send + 'static,
    {
        let settings = Arc::new(ArcSwap::from_pointee(initial));
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(AudioStats::default());

        let (frame_tx, frame_rx) = crossbeam_channel::bounded(2);
        let (sample_tx, sample_rx) = crossbeam_channel::bounded::<(Vec<f32>, Instant)>(32);
        let (ctrl_tx, ctrl_rx) = crossbeam_channel::unbounded::<ControlMsg>();
        let (info_tx, info_rx) = crossbeam_channel::bounded(1);
        let (tap_tx, tap_rx) = crossbeam_channel::unbounded::<Instant>();

        let sample_rate_shared = Arc::new(AtomicU32::new(48_000f32.to_bits()));

        {
            let settings = settings.clone();
            let stop = stop.clone();
            let sr = sample_rate_shared.clone();
            let stats = stats.clone();
            thread::spawn(move || {
                capture_loop(settings, sample_tx, ctrl_tx, info_tx, sr, stats, stop);
            });
        }

        let (device_name, sample_rate) = info_rx.recv()
            .unwrap_or_else(|_| ("Unknown".to_string(), 48000.0));
        sample_rate_shared.store(sample_rate.to_bits(), Ordering::Relaxed);

        {
            let settings = settings.clone();
            let stop = stop.clone();
            let sr = sample_rate_shared.clone();
            let stats = stats.clone();
            thread::spawn(move || {
                analysis_loop(
                    settings, sample_rx, ctrl_rx, tap_rx, frame_tx,
                    sr, stats, stop, on_frame,
                );
            });
        }

        Self { device_name, frame_rx, settings, stats, tap_tx, stop }
    }

    pub fn shutdown(&self) { self.stop.store(true, Ordering::Relaxed); }

    /// Register a tap (for tap-tempo). Call once per physical tap.
    pub fn tap(&self) {
        let _ = self.tap_tx.try_send(Instant::now());
    }

    pub fn tap_sender(&self) -> Sender<Instant> { self.tap_tx.clone() }
}

// ─────────────────────────────────────────────────────────────
// Capture
// ─────────────────────────────────────────────────────────────

#[derive(Debug)]
enum ControlMsg { ResetDcBlocker }

#[derive(Debug)]
enum SessionEnd { Stopped, Disconnected, Invalidated }

fn capture_loop(
    settings: Arc<ArcSwap<AudioSettings>>,
    sample_tx: Sender<(Vec<f32>, Instant)>,
    ctrl_tx: Sender<ControlMsg>,
    info_tx: Sender<(String, f32)>,
    sample_rate_shared: Arc<AtomicU32>,
    stats: Arc<AudioStats>,
    stop: Arc<AtomicBool>,
) {
    let _ = initialize_mta();
    let mut sent_info = false;
    let mut backoff = Duration::from_millis(500);

    loop {
        if stop.load(Ordering::Relaxed) { return; }
        match capture_session(
            &settings, &sample_tx, &ctrl_tx, &info_tx,
            &sample_rate_shared, &stats, &stop, &mut sent_info,
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
    sample_tx: &Sender<(Vec<f32>, Instant)>,
    ctrl_tx: &Sender<ControlMsg>,
    info_tx: &Sender<(String, f32)>,
    sample_rate_shared: &Arc<AtomicU32>,
    stats: &Arc<AudioStats>,
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

    stats.sample_rate_bits.store(sample_rate.to_bits(), Ordering::Relaxed);
    stats.channels.store(channels.min(u16::MAX as usize) as u16, Ordering::Relaxed);
    stats.bits_per_sample.store(bits.min(u16::MAX as usize) as u16, Ordering::Relaxed);

    if bits != 32 {
        eprintln!("WASAPI: non-float mix format ({bits} bits). Audio will be garbled.");
    }
    let samples_per_frame = (blockalign / 4).max(1).max(channels);

    let (_, min_time) = audio_client.get_periods().unwrap_or((100_000, 100_000));
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
        let _ = info_tx.send((device_name.clone(), sample_rate));
        *sent_info = true;
    }
    sample_rate_shared.store(sample_rate.to_bits(), Ordering::Relaxed);

    let mut sample_buffer: Vec<f32> = Vec::new();
    let mut mono: Vec<f32> = Vec::with_capacity(4096);
    let mut last_device_check = Instant::now();

    loop {
        if stop.load(Ordering::Relaxed) {
            let _ = audio_client.stop_stream();
            return SessionEnd::Stopped;
        }

        // ── auto-follow default device (feature 25) ──
        if last_device_check.elapsed() > DEVICE_POLL_INTERVAL {
            last_device_check = Instant::now();
            if let Ok(current) = get_default_device(&Direction::Render) {
                let current_name = current.get_friendlyname().unwrap_or_default();
                if current_name != device_name {
                    eprintln!("WASAPI: default device changed to '{current_name}'");
                    let _ = audio_client.stop_stream();
                    return SessionEnd::Invalidated;
                }
            }
        }

        let waited_at = Instant::now();
        if h_event.wait_for_event(250).is_err()
            && waited_at.elapsed() < Duration::from_millis(20)
        {
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

            if flags.data_discontinuity { let _ = ctrl_tx.try_send(ControlMsg::ResetDcBlocker); }
            if flags.silent { sample_buffer[..needed].fill(0.0); }

            let valid_samples =
                (frames_read as usize * samples_per_frame).min(sample_buffer.len());
            let samples = &sample_buffer[..valid_samples];

            let mix_mode = settings.load().channel_mix;
            mono.clear();
            mono.reserve(valid_samples / channels.max(1));
            for chunk in samples.chunks_exact(channels.max(1)) {
                mono.push(downmix(chunk, mix_mode));
            }
            if mono.is_empty() { continue; }
            let out = std::mem::replace(&mut mono, Vec::with_capacity(4096));
            let captured_at = Instant::now();
            stats.frames_captured.fetch_add(1, Ordering::Relaxed);
            stats.capture_time_us
                .store(waited_at.elapsed().as_micros() as u64, Ordering::Relaxed);
            match sample_tx.send_timeout((out, captured_at), Duration::from_millis(50)) {
                Ok(()) => {}
                Err(SendTimeoutError::Timeout(_)) => {
                    stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                }
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
    sample_rx: Receiver<(Vec<f32>, Instant)>,
    ctrl_rx: Receiver<ControlMsg>,
    tap_rx: Receiver<Instant>,
    frame_tx: Sender<BarFrame>,
    sample_rate_shared: Arc<AtomicU32>,
    stats: Arc<AudioStats>,
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
    let mut last_capture_instant = Instant::now();
    let mut idle = false;
    let mut last_sent_zero = false;
    let mut silent_frame_pending: Option<BarFrame> = None;

    loop {
        if stop.load(Ordering::Relaxed) { return; }

        while let Ok(msg) = ctrl_rx.try_recv() {
            match msg { ControlMsg::ResetDcBlocker => dc.reset() }
        }
        while let Ok(tap) = tap_rx.try_recv() {
            processor.record_tap(tap);
        }

        let new_sr = f32::from_bits(sample_rate_shared.load(Ordering::Relaxed));
        if (new_sr - sample_rate).abs() > 0.5 {
            eprintln!("Audio: sample rate changed {sample_rate} -> {new_sr}");
            sample_rate = new_sr;
            processor = AudioProcessor::new(sample_rate);
            dc = DcBlocker::new(sample_rate);
            history.clear();
        }

        let current = settings.load_full();
        stats.fft_size.store(current.fft_size as u32, Ordering::Relaxed);

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

            let (mut chunk, captured_at) = match sample_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(c) => c,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return,
            };
            last_capture_instant = captured_at;
            if peak_abs(&chunk) <= LOUD_THRESHOLD { continue; }
            idle = false;
            let now = Instant::now();
            last_loud = now; last_chunk = now; last_tick = now;
            if current.dc_block { dc.apply(&mut chunk); }
            history.clear();
            history.extend_from_slice(&chunk);
        }

        let poll_period = Duration::from_secs_f32(1.0 / current.poll_rate_hz.max(1.0));
        let analysis_t0 = Instant::now();

        let mut got_chunk = false;
        while let Ok((mut chunk, captured_at)) = sample_rx.try_recv() {
            got_chunk = true;
            last_capture_instant = captured_at;
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

        if let Some(frame) = processor.process(&history, &current, dt, last_capture_instant) {
            if frame.beat {
                stats.onset_count.fetch_add(1, Ordering::Relaxed);
            }
            let is_zero = frame.values.iter().all(|&v| v == 0.0)
                && frame.peaks.iter().all(|&v| v == 0.0);
            let should_send = !(is_zero && last_sent_zero) || frame.beat;
            if should_send {
                match frame_tx.try_send(frame) {
                    Ok(()) => { last_sent_zero = is_zero; on_frame(); }
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => return,
                }
            }
        }

        let elapsed_us = analysis_t0.elapsed().as_micros() as u64;
        stats.analysis_time_us.store(elapsed_us, Ordering::Relaxed);

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

struct OnsetDetector {
    prev_mags: Vec<f32>,
    band_flux_history: [VecDeque<f32>; 3],
    flux_history_total: VecDeque<f32>,
    band_armed: [bool; 3],
    total_armed: bool,
    last_fire: [Option<f64>; 3],
}

impl OnsetDetector {
    fn new(n_bins: usize) -> Self {
        Self {
            prev_mags: vec![0.0; n_bins],
            band_flux_history: [VecDeque::with_capacity(240), VecDeque::with_capacity(240), VecDeque::with_capacity(240)],
            flux_history_total: VecDeque::with_capacity(240),
            band_armed: [true; 3],
            total_armed: true,
            last_fire: [None; 3],
        }
    }

    fn reset(&mut self) {
        for v in &mut self.prev_mags { *v = 0.0; }
        for h in &mut self.band_flux_history { h.clear(); }
        self.flux_history_total.clear();
        self.band_armed = [true; 3];
        self.total_armed = true;
        self.last_fire = [None; 3];
    }

    fn detect(
        &mut self,
        mags: &[f32],
        n_bins: usize,
        bin_width: f32,
        poll_rate_hz: f32,
        sensitivity: f32,
        threshold_mult: f32,
        time_elapsed: f64,
    ) -> (Onsets, bool) {
        // Bands in Hz: kick 20-150, snare 150-2000, hat 6000-12000
        let band_bins = [
            (
                ((20.0 / bin_width).max(1.0) as usize).min(n_bins),
                ((150.0 / bin_width).ceil() as usize).min(n_bins),
            ),
            (
                ((150.0 / bin_width) as usize).min(n_bins),
                ((2000.0 / bin_width).ceil() as usize).min(n_bins),
            ),
            (
                ((6000.0 / bin_width) as usize).min(n_bins),
                ((12000.0 / bin_width).ceil() as usize).min(n_bins),
            ),
        ];

        let mut total_flux = 0.0f32;
        let mut band_flux = [0.0f32; 3];
        for k in 1..n_bins {
            let delta = (mags[k] - self.prev_mags[k]).max(0.0);
            total_flux += delta;
            for b in 0..3 {
                let (lo, hi) = band_bins[b];
                if k >= lo && k < hi { band_flux[b] += delta; }
            }
            self.prev_mags[k] = mags[k];
        }

        let hist_len = (poll_rate_hz * 1.5).max(30.0) as usize;
        let min_hist = (poll_rate_hz * 0.5).max(15.0) as usize;
        while self.flux_history_total.len() >= hist_len { self.flux_history_total.pop_front(); }
        self.flux_history_total.push_back(total_flux);
        for b in 0..3 {
            while self.band_flux_history[b].len() >= hist_len { self.band_flux_history[b].pop_front(); }
            self.band_flux_history[b].push_back(band_flux[b]);
        }

        let mut onsets = Onsets {
            kick: false, snare: false, hat: false,
            flux: 0.0,
        };

        if self.flux_history_total.len() < min_hist {
            return (onsets, false);
        }

        let sens = sensitivity.clamp(0.0, 1.0);
        let mult = threshold_mult.max(1.0) * (1.0 - sens * 0.5);
        let min_gap = MIN_BEAT_GAP_S;

        for b in 0..3 {
            let h = &self.band_flux_history[b];
            let avg = h.iter().sum::<f32>() / h.len() as f32;
            let thr = avg * mult;
            let above = band_flux[b] > thr && band_flux[b] > avg * 1.05;

            if band_flux[b] < thr * 0.9 { self.band_armed[b] = true; }

            let can_fire = above && self.band_armed[b]
                && self.last_fire[b].map(|t| time_elapsed - t > min_gap).unwrap_or(true);

            if can_fire {
                self.last_fire[b] = Some(time_elapsed);
                self.band_armed[b] = false;
                match b {
                    0 => onsets.kick = true,
                    1 => onsets.snare = true,
                    _ => onsets.hat = true,
                }
            }
        }

        let total_avg = self.flux_history_total.iter().sum::<f32>()
            / self.flux_history_total.len() as f32;
        let total_thr = total_avg * mult;
        let total_above = total_flux > total_thr && total_flux > total_avg * 1.05;
        if total_flux < total_thr * 0.9 { self.total_armed = true; }
        let total_fire = total_above && self.total_armed;
        if total_fire { self.total_armed = false; }

        onsets.flux = (total_flux / total_avg.max(1e-9)).min(4.0) / 4.0;
        (onsets, total_fire)
    }
}

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

    onset_detector: OnsetDetector,
    beat_times: VecDeque<f64>,
    bpm_raw: f32,
    bpm_multiplier: BpmMultiplier,
    phase: f32,
    beat_envelope: f32,
    beat_in: f32,
    time_elapsed: f64,
    next_beat_time: f64,
    last_fired_beat_time: f64,
    tap_times: VecDeque<Instant>,
    tap_override_until: Option<Instant>,

    band_levels: [f32; 4],
    waveform_buf: Vec<f32>,
    waveform_pos: usize,
    chroma_enabled: bool,
    chroma: [f32; 12],
    dominant_pitch_hz: f32,
    dominant_pitch_midi: f32,
    spectrum_row: Vec<f32>,
    last_rms: f32,
    last_peak: f32,
    last_onsets: Onsets,
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
            fft_size, num_bars, FrequencyScaling::Logarithmic,
            sample_rate, 30.0, 16_000.0,
        );
        let bin_centers_hz = compute_bar_centers_hz(
            num_bars, FrequencyScaling::Logarithmic, 30.0, 16_000.0,
        );
        let nyquist = fft_size / 2;

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
            onset_detector: OnsetDetector::new(nyquist),
            beat_times: VecDeque::with_capacity(MAX_BEATS),
            bpm_raw: 0.0,
            bpm_multiplier: BpmMultiplier::Normal,
            phase: 0.0,
            beat_envelope: 0.0,
            beat_in: 0.0,
            time_elapsed: 0.0,
            next_beat_time: f64::INFINITY,
            last_fired_beat_time: f64::NEG_INFINITY,
            tap_times: VecDeque::with_capacity(8),
            tap_override_until: None,
            band_levels: [0.0; 4],
            waveform_buf: vec![0.0; WAVEFORM_LEN],
            waveform_pos: 0,
            chroma_enabled: true,
            chroma: [0.0; 12],
            dominant_pitch_hz: 0.0,
            dominant_pitch_midi: 0.0,
            spectrum_row: vec![0.0; SPECTRUM_ROW_LEN],
            last_rms: 0.0,
            last_peak: 0.0,
            last_onsets: Onsets { kick: false, snare: false, hat: false, flux: 0.0 },
            bar_offset_db: Vec::new(),
            offset_centers: Vec::new(),
            offset_params: [f32::NAN; 4],
            offset_weighting: u8::MAX,
        }
    }

    fn record_tap(&mut self, t: Instant) {
        self.tap_times.push_back(t);
        while self.tap_times.len() > 8 { self.tap_times.pop_front(); }
        while let Some(&first) = self.tap_times.front() {
            if t.duration_since(first) > Duration::from_secs(3) {
                self.tap_times.pop_front();
            } else { break; }
        }
        if self.tap_times.len() >= 3 {
            let mut intervals: Vec<f64> = self.tap_times.iter()
                .zip(self.tap_times.iter().skip(1))
                .map(|(a, b)| b.duration_since(*a).as_secs_f64())
                .filter(|d| *d > 0.15 && *d < 2.0)
                .collect();
            if intervals.len() >= 2 {
                intervals.sort_by(|a, b| a.total_cmp(b));
                let median = intervals[intervals.len() / 2];
                if median > 0.15 && median < 2.0 {
                    self.bpm_raw = (60.0 / median) as f32;
                    self.tap_override_until = Some(t);
                    let period = 60.0 / (self.bpm_raw * self.bpm_multiplier.factor()).max(1.0);
                    self.phase = 0.0;
                    self.next_beat_time = self.time_elapsed + period as f64;
                    self.last_fired_beat_time = self.time_elapsed;
                }
            }
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
        { return; }
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
            if !s.fft_size.is_power_of_two() || s.fft_size < 256 { return; }
            let mut planner = RealFftPlanner::new();
            self.fft = planner.plan_fft_forward(s.fft_size);
            self.input_buffer = self.fft.make_input_vec();
            self.output_buffer = self.fft.make_output_vec();
            self.scratch_buffer = self.fft.make_scratch_vec();
            self.cached_fft_size = s.fft_size;
            self.cached_window_fn = s.window_function;
            self.window = make_window(s.window_function, s.fft_size);
            self.window_gain = compute_window_gain(&self.window);
            self.onset_detector = OnsetDetector::new(s.fft_size / 2);
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
            if s.num_bars == 0 || s.min_frequency_hz >= s.max_frequency_hz { return; }
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
            && self.last_rms < 1.0e-3
    }

    fn snap_to_silence(&mut self) {
        for v in self.bar_values.iter_mut() { *v = 0.0; }
        for v in self.bar_peaks.iter_mut() { *v = 0.0; }
        for v in self.bar_velocities.iter_mut() { *v = 0.0; }
        for v in self.peak_hold_remaining.iter_mut() { *v = 0.0; }
        self.frame_ring.fill(0.0);
        self.frame_ring_count = 0;
        self.beat_envelope = 0.0;
        self.agc_envelope = 1.0;
        self.last_rms = 0.0;
        self.last_peak = 0.0;
        self.band_levels = [0.0; 4];
        self.chroma = [0.0; 12];
        self.dominant_pitch_hz = 0.0;
        self.dominant_pitch_midi = 0.0;
        self.last_onsets = Onsets { kick: false, snare: false, hat: false, flux: 0.0 };
        self.onset_detector.reset();
        self.waveform_buf.fill(0.0);
        self.spectrum_row.fill(0.0);
        self.beat_times.clear();
        self.bpm_raw = 0.0;
        self.next_beat_time = f64::INFINITY;
        self.tap_override_until = None;
        self.tap_times.clear();
    }

    fn silent_frame(&self) -> BarFrame {
        BarFrame {
            values: vec![0.0; self.bar_values.len()],
            peaks: vec![0.0; self.bar_peaks.len()],
            beat: false, beat_predicted: false,
            beat_envelope: 0.0, beat_in: self.beat_in,
            bpm: self.effective_bpm(),
            phase: self.phase,
            onsets: Onsets { kick: false, snare: false, hat: false, flux: 0.0 },
            rms: 0.0, peak: 0.0, band_levels: [0.0; 4],
            chroma: [0.0; 12],
            dominant_pitch_hz: 0.0, dominant_pitch_midi: 0.0,
            waveform: vec![0.0; WAVEFORM_LEN],
            spectrum_row: vec![0.0; SPECTRUM_ROW_LEN],
            capture_instant: Instant::now(),
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

    fn effective_bpm(&self) -> f32 { self.bpm_raw * self.bpm_multiplier.factor() }

    fn process(
        &mut self,
        history: &[f32],
        s: &AudioSettings,
        dt: f32,
        capture_instant: Instant,
    ) -> Option<BarFrame> {
        self.update_initial_state(s);
        self.time_elapsed += dt as f64;
        self.bpm_multiplier = s.bpm_multiplier;
        self.chroma_enabled = s.chroma_enabled;

        let n_fft = self.cached_fft_size;
        if history.len() < n_fft {
            return Some(BarFrame {
                values: self.bar_values.clone(),
                peaks: self.bar_peaks.clone(),
                beat: false, beat_predicted: false,
                beat_envelope: self.beat_envelope, beat_in: self.beat_in,
                bpm: self.effective_bpm(), phase: self.phase,
                onsets: self.last_onsets.clone(),
                rms: self.last_rms, peak: self.last_peak,
                band_levels: self.band_levels,
                chroma: self.chroma,
                dominant_pitch_hz: self.dominant_pitch_hz,
                dominant_pitch_midi: self.dominant_pitch_midi,
                waveform: self.waveform_buf.clone(),
                spectrum_row: self.spectrum_row.clone(),
                capture_instant,
            });
        }
        let window_slice = &history[history.len() - n_fft..];

        let raw_rms = {
            let sum_sq: f32 = window_slice.iter().map(|x| x * x).sum();
            (sum_sq / n_fft as f32).sqrt()
        };
        let raw_peak = window_slice.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        self.last_rms = raw_rms;
        self.last_peak = raw_peak;

        let n_new = window_slice.len().min(WAVEFORM_LEN);
        let to_copy = &window_slice[window_slice.len() - n_new..];
        if n_new >= WAVEFORM_LEN {
            self.waveform_buf.copy_from_slice(to_copy);
            self.waveform_pos = 0;
        } else {
            for &sample in to_copy {
                self.waveform_buf[self.waveform_pos] = sample;
                self.waveform_pos = (self.waveform_pos + 1) % WAVEFORM_LEN;
            }
        }

        if s.agc_enabled && raw_rms > 1e-4 {
            let desired = (0.08 / raw_rms.max(1e-6)).clamp(0.01, 20.0);
            let coeff = if desired < self.agc_envelope { s.agc_attack.clamp(0.0001, 1.0) }
                        else { s.agc_release.clamp(0.0001, 1.0) };
            let alpha = 1.0 - (1.0 - coeff).powf(dt * 60.0);
            self.agc_envelope += (desired - self.agc_envelope) * alpha;
            self.agc_envelope = self.agc_envelope.clamp(0.01, 20.0);
        } else if !s.agc_enabled {
            self.agc_envelope = 1.0;
        }
        let agc_gain = self.agc_envelope.clamp(0.01, 20.0);
        let user_gain = s.raw_gain.max(0.01) * agc_gain;

        self.input_buffer.copy_from_slice(window_slice);
        for i in 0..n_fft { self.input_buffer[i] *= self.window[i]; }
        if self.fft.process_with_scratch(
            &mut self.input_buffer, &mut self.output_buffer, &mut self.scratch_buffer,
        ).is_err() { return None; }

        let num_bars = self.cached_num_bars;
        let nyquist = n_fft / 2;
        let norm = self.window_gain * user_gain;
        let bin_width = self.sample_rate / n_fft as f32;

        let band_hz = [(20.0, 100.0), (100.0, 500.0), (500.0, 2000.0), (2000.0, 8000.0)];
        for (b, (lo_hz, hi_hz)) in band_hz.iter().enumerate() {
            let lo = ((lo_hz / bin_width) as usize).min(nyquist);
            let hi = ((hi_hz / bin_width).ceil() as usize).min(nyquist);
            let mut sum = 0.0f32;
            for k in lo..hi.max(lo + 1) { sum += self.output_buffer[k].norm(); }
            let count = (hi - lo).max(1) as f32;
            self.band_levels[b] = ((sum / count) * norm).max(0.0).min(1.0);
        }

        for i in 0..SPECTRUM_ROW_LEN {
            let t = i as f32 / (SPECTRUM_ROW_LEN - 1) as f32;
            let freq = s.min_frequency_hz * (s.max_frequency_hz / s.min_frequency_hz).powf(t);
            let bin = ((freq / bin_width).round() as usize).min(nyquist.saturating_sub(1)).max(1);
            let mag = self.output_buffer[bin].norm() * norm;
            self.spectrum_row[i] = (20.0 * (mag + 1e-9).log10() + 91.0).max(0.0) / 91.0;
        }

        if self.chroma_enabled {
            self.compute_chroma_and_pitch(bin_width, nyquist);
        } else {
            self.chroma = [0.0; 12];
            self.dominant_pitch_hz = 0.0;
            self.dominant_pitch_midi = 0.0;
        }

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
                mag = self.output_buffer[k].norm() * (1.0 - t) + self.output_buffer[k1].norm() * t;
            } else {
                let mut sum = 0.0f32;
                for bin in start..end { sum += self.output_buffer[bin].norm(); }
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
            for i in 0..num_bars { self.smoothed[i] += self.frame_ring[offset + i]; }
        }
        let inv = 1.0 / self.frame_ring_count as f32;
        for v in self.smoothed.iter_mut() { *v *= inv; }

        if s.per_frame_normalization {
            let peak = self.smoothed.iter().cloned().fold(0.0f32, f32::max);
            if peak > 0.001 { for v in self.smoothed.iter_mut() { *v /= peak; } }
        }

        let smoothed = std::mem::take(&mut self.smoothed);
        self.update_ballistics(&smoothed, s, dt);
        self.smoothed = smoothed;

        let (beat, beat_predicted, phase, beat_in) = self.detect_rhythm(s, dt, bin_width, nyquist);

        Some(BarFrame {
            values: self.bar_values.clone(),
            peaks: self.bar_peaks.clone(),
            beat, beat_predicted,
            beat_envelope: self.beat_envelope,
            beat_in, bpm: self.effective_bpm(), phase,
            onsets: self.last_onsets.clone(),
            rms: raw_rms.min(1.0),
            peak: raw_peak.min(1.0),
            band_levels: self.band_levels,
            chroma: self.chroma,
            dominant_pitch_hz: self.dominant_pitch_hz,
            dominant_pitch_midi: self.dominant_pitch_midi,
            waveform: self.waveform_buf.clone(),
            spectrum_row: self.spectrum_row.clone(),
            capture_instant,
        })
    }

    fn compute_chroma_and_pitch(&mut self, bin_width: f32, nyquist: usize) {
        let mut chroma = [0.0f32; 12];
        let mut best_mag = 0.0f32;
        let mut best_bin = 0usize;
        for k in 1..nyquist {
            let freq = k as f32 * bin_width;
            if freq < 80.0 || freq > 4000.0 { continue; }
            let mag = self.output_buffer[k].norm();
            if mag < 1e-6 { continue; }
            let midi = 69.0 + 12.0 * (freq / 440.0).log2();
            let pc = ((midi.round() as i32).rem_euclid(12)) as usize;
            chroma[pc] += mag;
            if mag > best_mag { best_mag = mag; best_bin = k; }
        }
        let max = chroma.iter().cloned().fold(0.0f32, f32::max);
        if max > 1e-6 { for v in chroma.iter_mut() { *v /= max; } }
        self.chroma = chroma;
        if best_mag > 1e-5 && best_bin > 0 {
            let freq = best_bin as f32 * bin_width;
            self.dominant_pitch_hz = freq;
            self.dominant_pitch_midi = 69.0 + 12.0 * (freq / 440.0).log2();
        } else {
            self.dominant_pitch_hz = 0.0;
            self.dominant_pitch_midi = 0.0;
        }
    }

    fn update_ballistics(&mut self, inputs: &[f32], s: &AudioSettings, dt: f32) {
        let n = self.bar_values.len().min(inputs.len());
        let peak_hold_s = s.peak_hold_ms / 1000.0;

        for i in 0..n {
            let (atk_ms, dec_ms) = if s.per_band_ballistics {
                let f = self.bin_centers_hz.get(i).copied().unwrap_or(1000.0);
                if f < 250.0 { (s.bass_attack_ms, s.bass_decay_ms) }
                else if f < 4000.0 { (s.mid_attack_ms, s.mid_decay_ms) }
                else { (s.treble_attack_ms, s.treble_decay_ms) }
            } else { (s.attack_time_ms, s.decay_time_ms) };

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
                        if next <= input { self.bar_velocities[i] = 0.0; input } else { next }
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

    fn detect_rhythm(
        &mut self,
        s: &AudioSettings,
        dt: f32,
        bin_width: f32,
        nyquist: usize,
    ) -> (bool, bool, f32, f32) {
        let mut mags = vec![0.0f32; nyquist];
        for k in 1..nyquist { mags[k] = self.output_buffer[k].norm(); }

        let (onsets, any_fire) = self.onset_detector.detect(
            &mags, nyquist, bin_width,
            s.poll_rate_hz,
            s.beat_detection_sensitivity,
            s.onset_threshold_multiplier,
            self.time_elapsed,
        );

        let in_tap_override = self.tap_override_until
            .map(|u| u.elapsed() < Duration::from_secs(5))
            .unwrap_or(false);

        let since_last_real = self.time_elapsed
            - self.beat_times.back().copied().unwrap_or(f64::NEG_INFINITY);
        let is_real_beat = any_fire
            && !in_tap_override
            && since_last_real > MIN_BEAT_GAP_S;

        if is_real_beat {
            self.beat_times.push_back(self.time_elapsed);
            while self.beat_times.len() > MAX_BEATS { self.beat_times.pop_front(); }
            self.recompute_bpm();
        }
        while let Some(&t) = self.beat_times.front() {
            if self.time_elapsed - t > BEAT_MEMORY_S { self.beat_times.pop_front(); } else { break; }
        }
        if let Some(&last) = self.beat_times.back() {
            if self.time_elapsed - last > BEAT_TIMEOUT_S && !in_tap_override {
                self.beat_times.clear();
                self.bpm_raw = 0.0;
            }
        }

        let eff_bpm = self.effective_bpm();
        let mut fire_real = false;
        let mut fire_predicted = false;

        if is_real_beat {
            let period_s = if eff_bpm > 20.0 { 60.0 / eff_bpm } else { 0.5 };
            let phase_error = if self.phase < 0.5 { self.phase } else { 1.0 - self.phase };
            let on_beat = phase_error < 0.35;
            if on_beat && s.bpm_lock_enabled {
                let error = if self.phase < 0.5 { -self.phase } else { 1.0 - self.phase };
                self.phase = (self.phase + error * s.bpm_phase_smoothing.clamp(0.0, 1.0))
                    .rem_euclid(1.0);
            }
            self.last_fired_beat_time = self.time_elapsed;
            self.next_beat_time = self.time_elapsed + period_s as f64;
            if eff_bpm > 20.0 { self.phase = 0.0; }
            fire_real = true;
        }

        let eff_bpm = self.effective_bpm();
        if eff_bpm > 20.0 {
            let period_s = 60.0 / eff_bpm;
            self.phase = (self.phase + dt / period_s).rem_euclid(1.0);

            let now = self.time_elapsed;
            let grace = 0.08;
            if !fire_real && now > self.next_beat_time + grace {
                let since_fired = now - self.last_fired_beat_time;
                if since_fired > period_s as f64 * 0.6 {
                    fire_predicted = true;
                    self.last_fired_beat_time = now;
                    self.next_beat_time = now + period_s as f64;
                    self.phase = 0.0;
                } else {
                    self.next_beat_time = now + period_s as f64;
                }
            }
            self.beat_in = (1.0 - self.phase) * period_s - s.latency_offset_ms / 1000.0;
        } else {
            self.beat_in = 0.0;
        }

        if fire_real {
            self.beat_envelope = 1.0;
        } else if fire_predicted {
            self.beat_envelope = 0.7;
        } else if self.beat_envelope > 0.0 {
            let tau = (s.beat_envelope_decay_ms / 1000.0).max(0.001);
            self.beat_envelope *= (-dt / tau).exp();
            if self.beat_envelope < 0.001 { self.beat_envelope = 0.0; }
        }

        let phase_shifted = if eff_bpm > 20.0 && s.latency_offset_ms > 0.0 {
            let period_s = 60.0 / eff_bpm;
            let shift = s.latency_offset_ms / 1000.0 / period_s;
            (self.phase - shift).rem_euclid(1.0)
        } else {
            self.phase
        };

        self.last_onsets = onsets;

        (fire_real || fire_predicted, fire_predicted, phase_shifted, self.beat_in)
    }

    fn recompute_bpm(&mut self) {
        if self.beat_times.len() < 3 { return; }
        let min_iv = 60.0 / BPM_FOLD_MAX;
        let max_iv = 60.0 / BPM_FOLD_MIN;
        let mut iv: Vec<f64> = self.beat_times.iter()
            .zip(self.beat_times.iter().skip(1))
            .map(|(a, b)| b - a)
            .filter(|d| *d > 0.05)
            .map(|mut d| { while d < min_iv { d *= 2.0; } while d > max_iv { d *= 0.5; } d })
            .collect();
        if iv.len() < 2 { return; }
        iv.sort_by(|a, b| a.total_cmp(b));
        let median = iv[iv.len() / 2];
        let (sum, n) = iv.iter()
            .filter(|d| (**d - median).abs() <= median * 0.12)
            .fold((0.0f64, 0usize), |(s, n), d| (s + d, n + 1));
        if n == 0 { return; }
        let new_bpm = (60.0 / (sum / n as f64)) as f32;
        if self.bpm_raw <= 0.0 || (new_bpm - self.bpm_raw).abs() / self.bpm_raw > 0.12 {
            self.bpm_raw = new_bpm;
        } else {
            self.bpm_raw += (new_bpm - self.bpm_raw) * 0.3;
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

/// Mel conversion: 2595 * log10(1 + f/700)
fn hz_to_mel(f: f32) -> f32 { 2595.0 * (1.0 + f / 700.0).log10() }
fn mel_to_hz(m: f32) -> f32 { 700.0 * (10f32.powf(m / 2595.0) - 1.0) }

/// Bark approximation (Traunmüller): 26.81 * f / (1960 + f) - 0.53
fn hz_to_bark(f: f32) -> f32 { 26.81 * f / (1960.0 + f) - 0.53 }
fn bark_to_hz(z: f32) -> f32 {
    let z = z + 0.53;
    1960.0 * z / (26.81 - z)
}

fn make_bin_ranges(
    fft_size: usize, num_bars: usize, scaling: FrequencyScaling,
    sample_rate: f32, min_freq: f32, max_freq: f32,
) -> Vec<(usize, usize)> {
    let nyquist = fft_size / 2;
    let bin_width = sample_rate / fft_size as f32;
    let nyquist_hz = bin_width * nyquist as f32;
    let min_freq = min_freq.clamp(bin_width, nyquist_hz - bin_width);
    let max_freq = max_freq.clamp(min_freq + bin_width, nyquist_hz);
    let min_bin = ((min_freq / bin_width).floor() as usize).max(1);
    let max_bin = ((max_freq / bin_width).ceil() as usize).min(nyquist).max(min_bin + 1);

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
        FrequencyScaling::Mel => {
            let min_m = hz_to_mel(min_bin as f32 * bin_width);
            let max_m = hz_to_mel(max_bin as f32 * bin_width);
            for i in 0..num_bars {
                let m_lo = min_m + (max_m - min_m) * i as f32 / num_bars as f32;
                let m_hi = min_m + (max_m - min_m) * (i + 1) as f32 / num_bars as f32;
                let f_lo = mel_to_hz(m_lo);
                let f_hi = mel_to_hz(m_hi);
                let s = ((f_lo / bin_width).floor() as usize).max(min_bin).min(max_bin);
                let e = ((f_hi / bin_width).ceil() as usize).max(s + 1).min(max_bin);
                ranges.push((s, e));
            }
        }
        FrequencyScaling::Bark => {
            let min_b = hz_to_bark(min_bin as f32 * bin_width);
            let max_b = hz_to_bark(max_bin as f32 * bin_width);
            for i in 0..num_bars {
                let z_lo = min_b + (max_b - min_b) * i as f32 / num_bars as f32;
                let z_hi = min_b + (max_b - min_b) * (i + 1) as f32 / num_bars as f32;
                let f_lo = bark_to_hz(z_lo);
                let f_hi = bark_to_hz(z_hi);
                let s = ((f_lo / bin_width).floor() as usize).max(min_bin).min(max_bin);
                let e = ((f_hi / bin_width).ceil() as usize).max(s + 1).min(max_bin);
                ranges.push((s, e));
            }
        }
    }
    ranges
}

fn compute_bar_centers_hz(
    num_bars: usize, scaling: FrequencyScaling, min_freq: f32, max_freq: f32,
) -> Vec<f32> {
    if num_bars == 0 || max_freq <= min_freq { return Vec::new(); }
    let ratio = (max_freq / min_freq).powf(1.0 / num_bars as f32);
    (0..num_bars).map(|i| match scaling {
        FrequencyScaling::Linear => {
            let step = (max_freq - min_freq) / num_bars as f32;
            (min_freq + (i as f32 + 0.5) * step).max(20.0)
        }
        FrequencyScaling::Logarithmic => (min_freq * ratio.powf(i as f32 + 0.5)).max(20.0),
        FrequencyScaling::Mel => {
            let min_m = hz_to_mel(min_freq);
            let max_m = hz_to_mel(max_freq);
            let m = min_m + (max_m - min_m) * (i as f32 + 0.5) / num_bars as f32;
            mel_to_hz(m).max(20.0)
        }
        FrequencyScaling::Bark => {
            let min_b = hz_to_bark(min_freq);
            let max_b = hz_to_bark(max_freq);
            let z = min_b + (max_b - min_b) * (i as f32 + 0.5) / num_bars as f32;
            bark_to_hz(z).max(20.0)
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
    let shelf = biquad_db(f, FS,
        [1.535_124_86, -2.691_696_19, 1.198_392_81],
        [1.0, -1.690_659_29, 0.732_480_77]);
    let hp = biquad_db(f, FS, [1.0, -2.0, 1.0], [1.0, -1.990_047_45, 0.990_072_25]);
    shelf + hp
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_peak_is_one_at_centre() {
        let w = make_window(WindowFunction::Hann, 1024);
        assert!((w[512] - 1.0).abs() < 1e-3);
        assert!((w[0]).abs() < 1e-6);
        assert!((w[1023]).abs() < 1e-3);
    }

    #[test]
    fn rectangular_window_is_all_ones() {
        let w = make_window(WindowFunction::Rectangular, 256);
        assert!(w.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn bin_ranges_are_monotone_log() {
        let ranges = make_bin_ranges(
            2048, 64, FrequencyScaling::Logarithmic, 48000.0, 30.0, 16000.0);
        assert_eq!(ranges.len(), 64);
        for w in ranges.windows(2) {
            assert!(w[0].0 <= w[1].0, "start not monotone");
            assert!(w[1].0 >= w[0].1.saturating_sub(2), "gap");
        }
        assert!(ranges.iter().all(|(s, e)| e > s));
    }

    #[test]
    fn bin_ranges_cover_mel_and_bark() {
        for sc in [FrequencyScaling::Mel, FrequencyScaling::Bark] {
            let ranges = make_bin_ranges(2048, 40, sc, 48000.0, 30.0, 16000.0);
            assert!(ranges.iter().all(|(s, e)| e > s));
        }
    }

    #[test]
    fn bar_centers_log_ratio_is_constant() {
        let c = compute_bar_centers_hz(20, FrequencyScaling::Logarithmic, 30.0, 16000.0);
        let r0 = c[1] / c[0];
        for w in c.windows(2) {
            let r = w[1] / w[0];
            assert!((r - r0).abs() < 0.01 * r0);
        }
    }

    #[test]
    fn a_weighting_1khz_is_zero() {
        let db = a_weighting_db(1000.0);
        assert!(db.abs() < 0.5, "A at 1 kHz = {db}");
    }

    #[test]
    fn a_weighting_100hz_is_negative() {
        let db = a_weighting_db(100.0);
        assert!(db < -15.0 && db > -25.0, "A at 100 Hz = {db}");
    }

    #[test]
    fn k_weighting_is_bounded() {
        let db = k_weighting_db(1000.0);
        assert!(db.abs() < 5.0, "K at 1 kHz = {db}");
    }

    #[test]
    fn downmix_mono_of_identical_is_identity() {
        let chunk = [0.5f32, 0.5];
        assert!((downmix(&chunk, ChannelMixMode::Mono) - 0.5).abs() < 1e-6);
        assert!((downmix(&chunk, ChannelMixMode::Mid) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn downmix_side_of_identical_is_zero() {
        let chunk = [0.5f32, 0.5];
        assert!(downmix(&chunk, ChannelMixMode::Side).abs() < 1e-6);
    }

    #[test]
    fn downmix_side_of_opposite_is_half() {
        let chunk = [0.5f32, -0.5];
        assert!((downmix(&chunk, ChannelMixMode::Side) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dc_blocker_decays_dc() {
        let mut dc = DcBlocker::new(48000.0);
        let mut buf = vec![1.0f32; 48000];
        dc.apply(&mut buf);
        assert!(buf[47999].abs() < 0.1);
    }

    #[test]
    fn recompute_bpm_from_known_intervals() {
        let mut p = AudioProcessor::new(48000.0);
        for t in [1.0, 1.5, 2.0, 2.5, 3.0] {
            p.beat_times.push_back(t);
        }
        p.recompute_bpm();
        assert!((p.bpm_raw - 120.0).abs() < 2.0, "bpm = {}", p.bpm_raw);
    }

    #[test]
    fn smoothstep_endpoints() {
        assert!((smoothstep(0.0, 0.0, 1.0) - 0.0).abs() < 1e-6);
        assert!((smoothstep(1.0, 0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((smoothstep(0.5, 0.0, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mel_bark_roundtrip() {
        for f in [30.0f32, 100.0, 500.0, 2000.0, 8000.0] {
            let m = hz_to_mel(f);
            let f2 = mel_to_hz(m);
            assert!((f2 - f).abs() / f.max(1.0) < 0.01, "mel roundtrip {f} -> {f2}");
            let z = hz_to_bark(f);
            let f3 = bark_to_hz(z);
            assert!((f3 - f).abs() / f.max(1.0) < 0.05, "bark roundtrip {f} -> {f3}");
        }
    }

    #[test]
    fn barframe_lerp_midpoint() {
        let a = BarFrame {
            values: vec![0.0, 1.0], peaks: vec![0.0, 1.0],
            beat: false, beat_predicted: false, beat_envelope: 0.0, beat_in: 0.0,
            bpm: 120.0, phase: 0.0,
            onsets: Onsets { kick: false, snare: false, hat: false, flux: 0.0 },
            rms: 0.0, peak: 0.0, band_levels: [0.0; 4], chroma: [0.0; 12],
            dominant_pitch_hz: 0.0, dominant_pitch_midi: 0.0,
            waveform: vec![], spectrum_row: vec![0.0; 4],
            capture_instant: Instant::now(),
        };
        let b = BarFrame {
            values: vec![1.0, 0.0], peaks: vec![1.0, 0.0],
            ..a.clone()
        };
        let mid = a.lerp_toward(&b, 0.5);
        assert!((mid.values[0] - 0.5).abs() < 1e-6);
        assert!((mid.values[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn integration_sine_peak_at_correct_bar() {
        let sr = 48000.0;
        let mut p = AudioProcessor::new(sr);
        let mut settings = AudioSettings::default();
        settings.fft_size = 2048;
        settings.num_bars = 64;
        settings.min_frequency_hz = 30.0;
        settings.max_frequency_hz = 16000.0;
        settings.frequency_scaling = FrequencyScaling::Logarithmic;
        settings.dc_block = false;
        settings.agc_enabled = false;
        settings.noise_gate = 0.0;
        settings.attack_time_ms = 0.0;
        settings.decay_time_ms = 100.0;
        settings.average_size = 1;
        settings.per_frame_normalization = false;

        let freq = 1000.0f32;
        let n = 4096;
        let mut buf = vec![0.0f32; n];
        for i in 0..n {
            buf[i] = (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5;
        }

        let frame = p.process(&buf, &settings, 1.0 / 60.0, Instant::now()).unwrap();
        let peak_i = frame.values.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        let peak_hz = compute_bar_centers_hz(64, FrequencyScaling::Logarithmic, 30.0, 16000.0)[peak_i];
        assert!((peak_hz - freq).abs() / freq < 0.5, "peak at {peak_hz} Hz, expected ~{freq}");
    }
}
