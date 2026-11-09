#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use image::DynamicImage;
use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use arc_swap::ArcSwap;
use visualizer_core::{
    audio::{AudioEngine, BarFrame},
    color,
    config::{
        self as cfg,
        AppConfig, AudioPreset, BarInterpolation, BeatFocusMode, BlendMode, ChannelMixMode,
        ColorChangePreset, ColorMode, ColorPreset, ColorSource, CustomAudioPreset, DecayMode,
        FrequencyDirection, FrequencyScaling, GradientPreset, GradientSource, GradientStop,
        MotionShape, PalettePriority, StaticDistributionMode, StaticMultiPreset,
        TrackInfoConfig, TrackTextAlign, TrackTextColorMode, VerticalColorStyle,
        VisualizerDirection, WeightingCurve, WindowFunction,
    },
    musicbee::{self, TrackInfo},
};

use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyEvent, GlobalHotKeyManager,
};
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem},
    TrayIcon, TrayIconBuilder,
};

// ═══════════════════════════════════════════════════════════════
// Custom Icon Loader
// ═══════════════════════════════════════════════════════════════

fn load_icon() -> (egui::IconData, tray_icon::Icon) {
    let icon_bytes = include_bytes!("icon.png");
    let image = image::load_from_memory(icon_bytes).unwrap_or_else(|_| {
        image::DynamicImage::new_rgba8(1, 1)
    });
    let rgba = image.to_rgba8();
    let (w, h) = rgba.dimensions();
    let raw = rgba.into_raw();
    
    let egui_icon = egui::IconData {
        rgba: raw.clone(),
        width: w,
        height: h,
    };
    let tray = tray_icon::Icon::from_rgba(raw, w, h).unwrap();
    
    (egui_icon, tray)
}

// ═══════════════════════════════════════════════════════════════
// Small utilities
// ═══════════════════════════════════════════════════════════════

/// Wake the OVERLAY window. Never use `ctx.request_repaint()` from another
/// thread: in egui 0.27 it targets "the viewport currently being run", which
/// is the settings window while it is being painted (mouse moving over it).
/// That sent the wake-ups to the wrong window and froze the bars.
fn wake_root(ctx: &egui::Context) {
    ctx.request_repaint_of(egui::ViewportId::ROOT);
}

fn try_lock_state(m: &Mutex<SettingsState>) -> Option<std::sync::MutexGuard<'_, SettingsState>> {
    match m.try_lock() {
        Ok(g) => Some(g),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

fn tri01(t: f32) -> f32 { 1.0 - (2.0 * t.fract() - 1.0).abs() }

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn wipe_blend(pos: f32, boundary: f32, softness: f32, hard: bool) -> f32 {
    if hard {
        if pos < boundary { 0.0 } else { 1.0 }
    } else {
        smoothstep(boundary - softness, boundary + softness, pos)
    }
}

fn lerp_color(c1: egui::Color32, c2: egui::Color32, t: f32) -> egui::Color32 {
    let r = c1.r() as f32 * (1.0 - t) + c2.r() as f32 * t;
    let g = c1.g() as f32 * (1.0 - t) + c2.g() as f32 * t;
    let b = c1.b() as f32 * (1.0 - t) + c2.b() as f32 * t;
    let a = c1.a() as f32 * (1.0 - t) + c2.a() as f32 * t;
    egui::Color32::from_rgba_unmultiplied(r as u8, g as u8, b as u8, a as u8)
}

fn static_multi_vertical_color(colors: &[[u8; 3]], t: f32, blocks: bool) -> egui::Color32 {
    if colors.is_empty() { return egui::Color32::WHITE; }
    let n = colors.len();
    if n == 1 {
        let c = colors[0];
        return egui::Color32::from_rgb(c[0], c[1], c[2]);
    }
    if blocks {
        let idx = ((t * n as f32) as usize).min(n - 1);
        let c = colors[idx];
        egui::Color32::from_rgb(c[0], c[1], c[2])
    } else {
        let scaled = t.clamp(0.0, 1.0) * (n - 1) as f32;
        let i0 = scaled.floor() as usize;
        let i1 = (i0 + 1).min(n - 1);
        let f = scaled - i0 as f32;
        let a = colors[i0];
        let b = colors[i1];
        egui::Color32::from_rgb(
            (a[0] as f32 * (1.0 - f) + b[0] as f32 * f) as u8,
            (a[1] as f32 * (1.0 - f) + b[1] as f32 * f) as u8,
            (a[2] as f32 * (1.0 - f) + b[2] as f32 * f) as u8,
        )
    }
}

fn segment_rect(rect: egui::Rect, dir: VisualizerDirection, t0: f32, t1: f32) -> egui::Rect {
    let h = rect.height();
    let w = rect.width();
    match dir {
        VisualizerDirection::Up => egui::Rect::from_min_max(
            egui::pos2(rect.left(), rect.bottom() - h * t1),
            egui::pos2(rect.right(), rect.bottom() - h * t0),
        ),
        VisualizerDirection::Down => egui::Rect::from_min_max(
            egui::pos2(rect.left(), rect.top() + h * t0),
            egui::pos2(rect.right(), rect.top() + h * t1),
        ),
        VisualizerDirection::Right => egui::Rect::from_min_max(
            egui::pos2(rect.left() + w * t0, rect.top()),
            egui::pos2(rect.left() + w * t1, rect.bottom()),
        ),
        VisualizerDirection::Left => egui::Rect::from_min_max(
            egui::pos2(rect.right() - w * t1, rect.top()),
            egui::pos2(rect.right() - w * t0, rect.bottom()),
        ),
    }
}

fn segment_rounding(dir: VisualizerDirection, is_first: bool, is_last: bool, radius: f32) -> egui::Rounding {
    let mut r = egui::Rounding::ZERO;
    if !is_first && !is_last { return r; }
    match dir {
        VisualizerDirection::Up => {
            if is_last  { r.nw = radius; r.ne = radius; }
            if is_first { r.sw = radius; r.se = radius; }
        }
        VisualizerDirection::Down => {
            if is_first { r.nw = radius; r.ne = radius; }
            if is_last  { r.sw = radius; r.se = radius; }
        }
        VisualizerDirection::Right => {
            if is_first { r.nw = radius; r.sw = radius; }
            if is_last  { r.ne = radius; r.se = radius; }
        }
        VisualizerDirection::Left => {
            if is_first { r.ne = radius; r.se = radius; }
            if is_last  { r.nw = radius; r.sw = radius; }
        }
    }
    r
}

fn bar_vertex(dir: VisualizerDirection, perp: f32, anchor: f32, len: f32, t: f32) -> egui::Pos2 {
    match dir {
        VisualizerDirection::Up    => egui::pos2(perp, anchor - len * t),
        VisualizerDirection::Down  => egui::pos2(perp, anchor + len * t),
        VisualizerDirection::Right => egui::pos2(anchor + len * t, perp),
        VisualizerDirection::Left  => egui::pos2(anchor - len * t, perp),
    }
}

fn draw_peak_hold(
    painter: &egui::Painter,
    dir: VisualizerDirection,
    anchor: f32,
    perp: f32,
    peak_len: f32,
    bar_width: f32,
    color: egui::Color32,
    is_smooth: bool,
) {
    let pos = bar_vertex(dir, perp, anchor, peak_len, 1.0);
    if is_smooth {
        painter.circle_filled(pos, (bar_width / 4.0).clamp(1.0, 3.0), color);
    } else {
        let dash_size = match dir {
            VisualizerDirection::Up | VisualizerDirection::Down => egui::vec2(bar_width, 3.0),
            VisualizerDirection::Left | VisualizerDirection::Right => egui::vec2(3.0, bar_width),
        };
        let rect = egui::Rect::from_center_size(pos, dash_size);
        painter.rect_filled(rect, 1.0, color);
    }
}

#[derive(Clone, Copy)]
struct MotionSample {
    phase_a: f32, phase_b: f32, blend_t: f32, intensity: f32,
}
impl Default for MotionSample {
    fn default() -> Self { Self { phase_a: 0.0, phase_b: 0.0, blend_t: 0.0, intensity: 1.0 } }
}

fn random_color_index(i: usize, num_colors: usize) -> usize {
    (hash_u32(i as u32) as usize) % num_colors
}

fn stops_from_colors(colors: &[[u8; 3]]) -> Vec<GradientStop> {
    match colors.len() {
        0 => vec![GradientStop { color: [255, 255, 255], position: 0.0 }],
        1 => vec![GradientStop { color: colors[0], position: 0.0 }],
        n => colors.iter().enumerate()
            .map(|(i, c)| GradientStop { color: *c, position: i as f32 / (n - 1) as f32 })
            .collect(),
    }
}

fn compute_track_id(bytes: Option<&[u8]>, artist: &str, title: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    if let Some(b) = bytes { b.hash(&mut h); }
    artist.hash(&mut h);
    title.hash(&mut h);
    h.finish()
}

fn frequency_index(i: usize, base_num_bars: usize, dir: FrequencyDirection) -> usize {
    match dir {
        FrequencyDirection::LeftToRight => i,
        FrequencyDirection::RightToLeft => base_num_bars - 1 - i,
        FrequencyDirection::Mirrored => if i < base_num_bars { i } else { 2 * base_num_bars - 1 - i },
    }
}

// ═══════════════════════════════════════════════════════════════
// Fonts
// ═══════════════════════════════════════════════════════════════

const CUSTOM_FONT_FAMILY: &str = "custom_track";
const CUSTOM_FONT_KEY: &str = "track_custom";

fn custom_font_family() -> egui::FontFamily {
    egui::FontFamily::Name(CUSTOM_FONT_FAMILY.into())
}

fn list_system_fonts() -> Vec<(String, String)> {
    let fonts_dir = std::path::Path::new("C:\\Windows\\Fonts");
    let mut result = Vec::new();
    if let Ok(entries) = std::fs::read_dir(fonts_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ext == "ttf" || ext == "otf" {
                    if let Some(name) = path.file_name() {
                        result.push((
                            name.to_string_lossy().to_string(),
                            path.to_string_lossy().to_string(),
                        ));
                    }
                }
            }
        }
    }
    result.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
    result
}

// ═══════════════════════════════════════════════════════════════
// Palette cache
// ═══════════════════════════════════════════════════════════════

struct PaletteCacheEntry {
    track_id: u64, palette_size: u8, variants: u8, priority: PalettePriority,
    colors: Vec<[u8; 3]>,
}
#[derive(Default)]
struct PaletteCache { entry: Option<PaletteCacheEntry> }
impl PaletteCache {
    fn get(&self, track_id: u64, size: u8, variants: u8, priority: PalettePriority) -> Option<&[[u8; 3]]> {
        self.entry.as_ref()
            .filter(|e| e.track_id == track_id && e.palette_size == size
                    && e.variants == variants && e.priority == priority)
            .map(|e| e.colors.as_slice())
    }
    fn store(&mut self, track_id: u64, size: u8, variants: u8, priority: PalettePriority, colors: Vec<[u8; 3]>) {
        self.entry = Some(PaletteCacheEntry { track_id, palette_size: size, variants, priority, colors });
    }
}

#[derive(Default, Clone)]
struct ActivePalettes {
    static_single_colors: Vec<[u8; 3]>,
    static_multi_colors: Vec<[u8; 3]>,
    gradient_colors: Vec<[u8; 3]>,
    gradient_stops: Vec<GradientStop>,
    animated_colors: Vec<[u8; 3]>,
}

// ═══════════════════════════════════════════════════════════════
// Fingerprint helpers
// ═══════════════════════════════════════════════════════════════

fn hash_color_source<H: std::hash::Hasher>(s: &ColorSource, h: &mut H) {
    use std::hash::Hash;
    match s {
        ColorSource::Static(v) => { 0u8.hash(h); v.hash(h); }
        ColorSource::Dynamic { palette_size, variants_per_color, priority } => {
            1u8.hash(h);
            palette_size.hash(h);
            variants_per_color.hash(h);
            priority.hash(h);
        }
    }
}
fn hash_gradient_source<H: std::hash::Hasher>(s: &GradientSource, h: &mut H) {
    use std::hash::Hash;
    match s {
        GradientSource::Static(stops) => {
            0u8.hash(h);
            for stop in stops {
                stop.color.hash(h);
                stop.position.to_bits().hash(h);
            }
        }
        GradientSource::Dynamic { palette_size, variants_per_color, priority } => {
            1u8.hash(h);
            palette_size.hash(h);
            variants_per_color.hash(h);
            priority.hash(h);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Settings UI helpers
// ═══════════════════════════════════════════════════════════════

fn accent() -> egui::Color32 { egui::Color32::from_rgb(0xB7, 0x41, 0x0E) }

fn section_header(ui: &mut egui::Ui, title: &str) {
    ui.add_space(8.0);
    ui.label(egui::RichText::new(title).size(16.0).strong().color(accent()));
    ui.add_space(2.0);
    ui.separator();
    ui.add_space(4.0);
}

fn reset_btn<F: FnOnce()>(ui: &mut egui::Ui, changed: bool, on_reset: F) {
    let text = egui::RichText::new("Reset").size(12.0);
    if ui.add_enabled(changed, egui::Button::new(text).small()).clicked() {
        on_reset();
    }
}

fn help_text(ui: &mut egui::Ui, show: bool, text: &str) {
    if show {
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(format!("ℹ {}", text))
                .size(12.0)
                .color(egui::Color32::from_gray(150))
                .italics()
        );
        ui.add_space(6.0);
    }
}

fn slider_f32(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str, default: f32) {
    ui.horizontal(|ui| {
        let r = ui.add(egui::Slider::new(value, range).text(label));
        if r.secondary_clicked() { *value = default; }
        reset_btn(ui, (*value - default).abs() > 1e-6, || *value = default);
    });
}

fn slider_f32_scale(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str, default: f32) {
    ui.horizontal(|ui| {
        let r = ui.add(egui::Slider::new(value, range).text(label)
            .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)));
        if r.secondary_clicked() { *value = default; }
        reset_btn(ui, (*value - default).abs() > 1e-6, || *value = default);
    });
}

fn slider_u8(ui: &mut egui::Ui, value: &mut u8, label: &str, default: u8) {
    ui.horizontal(|ui| {
        let mut v = *value as i32;
        let r = ui.add(egui::Slider::new(&mut v, 0..=255).text(label));
        if v != *value as i32 { *value = v.clamp(0, 255) as u8; }
        if r.secondary_clicked() { *value = default; }
        reset_btn(ui, *value != default, || *value = default);
    });
}

fn checkbox_reset(ui: &mut egui::Ui, value: &mut bool, label: &str, default: bool) {
    ui.horizontal(|ui| {
        let r = ui.checkbox(value, label);
        if r.secondary_clicked() { *value = default; }
        reset_btn(ui, *value != default, || *value = default);
    });
}

fn color_edit_reset(ui: &mut egui::Ui, value: &mut [u8; 3], label: &str, default: [u8; 3]) {
    ui.horizontal(|ui| {
        ui.label(label);
        let r = ui.color_edit_button_srgb(value);
        if r.secondary_clicked() { *value = default; }
        reset_btn(ui, *value != default, || *value = default);
    });
}

fn combo_reset<T: PartialEq + Copy + 'static>(
    ui: &mut egui::Ui,
    value: &mut T,
    default: T,
    label: &str,
    id: &str,
    width: f32,
    options: &[T],
    name: fn(T) -> &'static str,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_source(id)
            .selected_text(name(*value))
            .width(width)
            .show_ui(ui, |ui| {
                for &opt in options {
                    ui.selectable_value(value, opt, name(opt));
                }
            });
        reset_btn(ui, *value != default, || *value = default);
    });
}

// ═══════════════════════════════════════════════════════════════
// Settings tabs
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Visualizer, Colors, Audio, TrackInfo, Appearance, Performance,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorSubTab {
    Single, Multi, Gradient, Animated,
}

const SEARCH_INDEX: &[(SettingsTab, &str, &[&str])] = &[
    (SettingsTab::Visualizer,  "Direction",          &["direction", "up", "down", "left", "right"]),
    (SettingsTab::Visualizer,  "Frequency Layout",   &["frequency", "mirrored", "layout", "scale"]),
    (SettingsTab::Visualizer,  "Bar Shape",          &["bar", "shape", "step", "smooth", "interpolation"]),
    (SettingsTab::Visualizer,  "Bars & Position",    &["bar", "width", "gap", "height", "offset", "position"]),
    (SettingsTab::Visualizer,  "Outlines",           &["outline", "border", "stroke"]),
    (SettingsTab::Colors,      "Color Source",       &["color", "palette", "album", "dynamic"]),
    (SettingsTab::Colors,      "Distribution",       &["blocks", "sequential", "random", "vertical"]),
    (SettingsTab::Colors,      "Gradient",           &["gradient", "stops", "angle"]),
    (SettingsTab::Colors,      "Motion",             &["motion", "animated", "shape"]),
    (SettingsTab::Audio,       "Presets",            &["preset", "audio", "custom"]),
    (SettingsTab::Audio,       "Signal Processing",  &["fft", "window", "frequency", "channel", "weighting", "smoothing", "dc", "harmonic"]),
    (SettingsTab::Audio,       "Amplitude",          &["sensitivity", "gain", "gamma", "noise", "gate", "db", "dynamic", "agc"]),
    (SettingsTab::Audio,       "Equalizer",          &["eq", "bass", "mid", "treble", "tilt", "spectrum"]),
    (SettingsTab::Audio,       "Ballistics",         &["attack", "decay", "ballistic", "gravity", "peak", "average"]),
    (SettingsTab::Audio,       "Rhythm",             &["beat", "bpm", "rhythm", "tempo", "phase"]),
    (SettingsTab::TrackInfo,   "Cover Art",          &["cover", "art", "album", "opacity"]),
    (SettingsTab::TrackInfo,   "Text",               &["title", "artist", "font", "text", "spacing"]),
    (SettingsTab::TrackInfo,   "Shadow",             &["shadow", "glow", "blur"]),
    (SettingsTab::TrackInfo,   "Outline",            &["outline", "border"]),
    (SettingsTab::Appearance,  "Background",         &["background", "bg", "color", "opacity"]),
    (SettingsTab::Appearance,  "UI",                 &["scale", "size", "help", "launch"]),
    (SettingsTab::Performance, "Performance",        &["fps", "vsync", "performance", "frame"]),
];

// ═══════════════════════════════════════════════════════════════
// App
// ═══════════════════════════════════════════════════════════════

/// Commands forwarded from the tray-menu / hotkey listener threads to the UI.
enum UiCmd {
    OpenSettings,
    ToggleSettings,
    Quit,
}

/// Everything the settings window owns. It lives behind a mutex because the
/// settings window is a *deferred* viewport: it repaints on its own (only on
/// input) instead of being rebuilt inside the overlay's frame loop.
struct SettingsState {
    config: AppConfig,
    audio_dirty: bool,
    close_requested: bool,
    last_fingerprint: u64,
    last_fp_check: Instant,
    old_settings_scale: f32,

    new_preset_name: String,
    new_audio_preset_name: String,
    last_decay_mode: DecayMode,
    settings_tab: SettingsTab,
    color_subtab: ColorSubTab,
    settings_search: String,
    pending_audio_preset_delete: Option<usize>,

    // Read-only mirrors pushed by the overlay.
    active_palettes: ActivePalettes,
    last_bpm: f32,
    current_fps: f32,
    system_fonts: Vec<(String, String)>,
}

fn lock_state(m: &Mutex<SettingsState>) -> std::sync::MutexGuard<'_, SettingsState> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Cheap change detector for the config (AppConfig has no PartialEq).
/// Only evaluated when the settings window actually received input.
fn config_fingerprint(c: &AppConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    match toml::to_string(c) {
        Ok(s) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }
        // Can't serialise → report "changed" so we never miss an edit.
        Err(_) => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
    }
}

struct VisualizerApp {
    config: AppConfig,
    current_bars: Vec<f32>,
    current_peaks: Vec<f32>,
    last_bpm: f32,
    audio_rx: crossbeam_channel::Receiver<BarFrame>,
    /// true while the overlay is animating and re-arms its own repaints
    /// (audio threads then must NOT wake it, so the FPS cap is respected).
    ui_active: Arc<AtomicBool>,
    palette_mirrored: bool,
    audio_settings: Arc<ArcSwap<visualizer_core::config::AudioSettings>>,
    should_quit: Arc<AtomicBool>,
    current_fps: f32,
    is_passthrough_set: bool,
    last_frame_time: Instant,
    track_rx: crossbeam_channel::Receiver<TrackInfo>,
    track_info: Option<TrackInfo>,
    cover_texture: Option<egui::TextureHandle>,
    cover_image: Option<DynamicImage>,
    track_id: u64,
    active_palettes: ActivePalettes,
    static_single_cache: PaletteCache,
    static_multi_cache: PaletteCache,
    gradient_cache: PaletteCache,
    animated_cache: PaletteCache,
    last_palette_hash: u64,
    loaded_font_path: Option<String>,
    smtc_session_count: Arc<AtomicUsize>,

    // settings window plumbing
    shared: Arc<Mutex<SettingsState>>,
    settings_open: Arc<AtomicBool>,
    config_gen: Arc<AtomicU64>,
    seen_gen: u64,
    was_settings_open: bool,
    settings_icon: egui::IconData,
    ui_cmd_rx: crossbeam_channel::Receiver<UiCmd>,
    
    // system
    _tray_icon: TrayIcon,
    _hotkey_manager: GlobalHotKeyManager,
}

impl VisualizerApp {
    fn new(ctx: egui::Context) -> Self {
        let config = AppConfig::load();

        // Audio thread wakes the UI whenever it publishes a frame.
        let audio_ctx = ctx.clone();
        let ui_active = Arc::new(AtomicBool::new(false));
        let ui_active_audio = ui_active.clone();
        let engine = AudioEngine::new(config.audio.clone(), move || {
            if !ui_active_audio.load(Ordering::Acquire) {
                wake_root(&audio_ctx);
            }
        });
        let should_quit = Arc::new(AtomicBool::new(false));
        musicbee::log_smtc("APP: Initializing VisualizerApp.");
        let audio_settings = engine.settings.clone();

        // SMTC thread wakes the UI when the track changes.
        let (track_tx, track_rx) = crossbeam_channel::unbounded();
        let smtc_session_count = Arc::new(AtomicUsize::new(0));
        let smtc_ctx = ctx.clone();
        musicbee::spawn_smtc_listener(
            track_tx,
            should_quit.clone(),
            smtc_session_count.clone(),
            move || wake_root(&smtc_ctx),
        );

        let (egui_icon, tray_icon_img) = load_icon();

        let tray_menu = Menu::new();
        let settings_item = MenuItem::new("Open Settings", true, None);
        let quit_item = MenuItem::new("Quit", true, None);
        tray_menu.append_items(&[&settings_item, &quit_item]).unwrap();
        let tray_icon = TrayIconBuilder::new()
            .with_menu(Box::new(tray_menu))
            .with_tooltip("Rust Music Visualizer")
            .with_icon(tray_icon_img)
            .build()
            .unwrap();

        let hotkey_manager = GlobalHotKeyManager::new().unwrap();
        let hotkey = HotKey::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyV);
        hotkey_manager.register(hotkey).unwrap();

        // ── Event listener threads: block on the OS event queues and wake the
        //    UI only when something really happened (no per-frame polling). ──
        let (ui_cmd_tx, ui_cmd_rx) = crossbeam_channel::unbounded::<UiCmd>();
        {
            let tx = ui_cmd_tx.clone();
            let wake = ctx.clone();
            let settings_id = settings_item.id().clone();
            let quit_id = quit_item.id().clone();
            std::thread::spawn(move || {
                let rx = MenuEvent::receiver();
                while let Ok(event) = rx.recv() {
                    let cmd = if event.id == settings_id { Some(UiCmd::OpenSettings) }
                              else if event.id == quit_id { Some(UiCmd::Quit) }
                              else { None };
                    if let Some(cmd) = cmd {
                        let _ = tx.send(cmd);
                        wake_root(&wake);
                    }
                }
            });
        }
        {
            let tx = ui_cmd_tx.clone();
            let wake = ctx.clone();
            std::thread::spawn(move || {
                let rx = GlobalHotKeyEvent::receiver();
                while let Ok(event) = rx.recv() {
                    if event.state == global_hotkey::HotKeyState::Pressed {
                        let _ = tx.send(UiCmd::ToggleSettings);
                        wake_root(&wake);
                    }
                }
            });
        }

        let num_bars = config.audio.num_bars;
        let settings_open = Arc::new(AtomicBool::new(config.open_settings_on_launch));

        let shared = Arc::new(Mutex::new(SettingsState {
            config: config.clone(),
            audio_dirty: false,
            close_requested: false,
            last_fingerprint: config_fingerprint(&config),
            last_fp_check: Instant::now(),
            old_settings_scale: config.settings_scale,
            new_preset_name: String::new(),
            new_audio_preset_name: String::new(),
            last_decay_mode: config.audio.decay_mode,
            settings_tab: SettingsTab::Visualizer,
            color_subtab: ColorSubTab::Single,
            settings_search: String::new(),
            pending_audio_preset_delete: None,
            active_palettes: ActivePalettes::default(),
            last_bpm: 0.0,
            current_fps: 0.0,
            system_fonts: list_system_fonts(),
        }));

        Self {
            current_bars: vec![0.0; num_bars],
            current_peaks: vec![0.0; num_bars],
            last_bpm: 0.0,
            audio_rx: engine.frame_rx,
            ui_active,
            palette_mirrored: true,
            audio_settings,
            config,
            should_quit,
            current_fps: 0.0,
            is_passthrough_set: false,
            last_frame_time: Instant::now(),
            track_rx,
            track_info: None,
            cover_texture: None,
            cover_image: None,
            track_id: 0,
            active_palettes: ActivePalettes::default(),
            static_single_cache: PaletteCache::default(),
            static_multi_cache: PaletteCache::default(),
            gradient_cache: PaletteCache::default(),
            animated_cache: PaletteCache::default(),
            last_palette_hash: 0,
            loaded_font_path: Some("__init__".to_string()),
            smtc_session_count,
            shared,
            settings_open,
            config_gen: Arc::new(AtomicU64::new(0)),
            seen_gen: 0,
            was_settings_open: false,
            settings_icon: egui_icon,
            ui_cmd_rx,
            _tray_icon: tray_icon,
            _hotkey_manager: hotkey_manager,
        }
    }

    fn poll_track_updates(&mut self, ctx: &egui::Context) {
        while let Ok(info) = self.track_rx.try_recv() {
            if let Some(bytes) = &info.cover_art {
                if let Some((colors, img)) = color::extract_colors_from_bytes(bytes) {
                    let _ = colors.dominant;
                    self.track_id = compute_track_id(Some(bytes), &info.artist, &info.title);
                    let rgba = img.to_rgba8();
                    let size = [rgba.width() as usize, rgba.height() as usize];
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                    self.cover_texture = Some(ctx.load_texture("cover_art", color_image, egui::TextureOptions::LINEAR));
                    self.cover_image = Some(img);
                }
            } else {
                self.cover_texture = None;
                self.cover_image = None;
                self.track_id = compute_track_id(None, &info.artist, &info.title);
            }
            if info.title.is_empty() && info.artist.is_empty() { self.track_info = None; }
            else { self.track_info = Some(info); }
        }
    }

    fn reload_custom_font(&mut self, ctx: &egui::Context) {
        let target = self.config.track_info.font_path.clone();
        let mut fonts = egui::FontDefinitions::default();
        let mut custom_family_names: Vec<String> = Vec::new();
        if let Some(ref path) = target {
            if let Ok(data) = std::fs::read(path) {
                fonts.font_data.insert(CUSTOM_FONT_KEY.to_owned(), egui::FontData::from_owned(data));
                custom_family_names.push(CUSTOM_FONT_KEY.to_owned());
            }
        }
        if custom_family_names.is_empty() {
            if let Some(defaults) = fonts.families.get(&egui::FontFamily::Proportional).cloned() {
                custom_family_names = defaults;
            }
        }
        fonts.families.insert(custom_font_family(), custom_family_names);
        ctx.set_fonts(fonts);
        self.loaded_font_path = target;
    }

    // ── Palettes ─────────────────────────────────────────────

    fn update_palettes(&mut self) {
        match self.config.color_mode {
            ColorMode::StaticSingle => self.update_static_single_palette(),
            ColorMode::StaticMultiple => self.update_static_multi_palette(),
            ColorMode::Gradient => self.update_gradient_palette(),
            ColorMode::Animated => self.update_animated_palette(),
        }
    }

    fn update_static_single_palette(&mut self) {
        let colors = match self.config.static_single_source.clone() {
            ColorSource::Static(colors) => vec![colors.first().copied().unwrap_or([255, 255, 255])],
            ColorSource::Dynamic { priority, .. } => {
                let tid = self.track_id;
                self.static_single_cache.get(tid, 1, 0, priority).map(|s| s.to_vec()).unwrap_or_else(|| {
                    let extracted = self.cover_image.as_ref()
                        .map(|img| color::extract_palette(img, 1, 0, priority))
                        .unwrap_or_else(|| vec![[255, 255, 255]]);
                    self.static_single_cache.store(tid, 1, 0, priority, extracted.clone());
                    extracted
                })
            }
        };
        self.active_palettes.static_single_colors = colors;
    }

    fn update_static_multi_palette(&mut self) {
        let colors = match self.config.static_multi_source.clone() {
            ColorSource::Static(colors) => if colors.is_empty() { vec![[255, 255, 255]] } else { colors },
            ColorSource::Dynamic { palette_size, variants_per_color, priority } => {
                let tid = self.track_id;
                self.static_multi_cache.get(tid, palette_size, variants_per_color, priority)
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| {
                        let extracted = self.cover_image.as_ref()
                            .map(|img| color::extract_palette(img, palette_size, variants_per_color, priority))
                            .unwrap_or_else(|| vec![[255, 255, 255]]);
                        self.static_multi_cache.store(tid, palette_size, variants_per_color, priority, extracted.clone());
                        extracted
                    })
            }
        };
        self.active_palettes.static_multi_colors = colors;
    }

    fn update_gradient_palette(&mut self) {
        let colors = match self.config.gradient_source.clone() {
            GradientSource::Static(stops) => stops.iter().map(|s| s.color).collect::<Vec<_>>(),
            GradientSource::Dynamic { palette_size, variants_per_color, priority } => {
                let tid = self.track_id;
                self.gradient_cache.get(tid, palette_size, variants_per_color, priority)
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| {
                        let extracted = self.cover_image.as_ref()
                            .map(|img| color::extract_palette(img, palette_size, variants_per_color, priority))
                            .unwrap_or_else(|| vec![[255, 255, 255]]);
                        self.gradient_cache.store(tid, palette_size, variants_per_color, priority, extracted.clone());
                        extracted
                    })
            }
        };
        self.active_palettes.gradient_stops = match &self.config.gradient_source {
            GradientSource::Static(stops) => stops.clone(),
            GradientSource::Dynamic { .. } => stops_from_colors(&colors),
        };
        self.active_palettes.gradient_colors = colors;
    }

    fn update_animated_palette(&mut self) {
        let colors = match self.config.animated_source.clone() {
            ColorSource::Static(colors) => if colors.is_empty() { vec![[255, 255, 255]] } else { colors },
            ColorSource::Dynamic { palette_size, variants_per_color, priority } => {
                let tid = self.track_id;
                self.animated_cache.get(tid, palette_size, variants_per_color, priority)
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| {
                        let extracted = self.cover_image.as_ref()
                            .map(|img| color::extract_palette(img, palette_size, variants_per_color, priority))
                            .unwrap_or_else(|| vec![[255, 255, 255]]);
                        self.animated_cache.store(tid, palette_size, variants_per_color, priority, extracted.clone());
                        extracted
                    })
            }
        };
        self.active_palettes.animated_colors = colors;
    }

    fn palette_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.track_id.hash(&mut h);
        std::mem::discriminant(&self.config.color_mode).hash(&mut h);
        hash_color_source(&self.config.static_single_source, &mut h);
        hash_color_source(&self.config.static_multi_source, &mut h);
        hash_gradient_source(&self.config.gradient_source, &mut h);
        hash_color_source(&self.config.animated_source, &mut h);
        h.finish()
    }

    // ── Color helpers ────────────────────────────────────────

    fn get_visualizer_text_color(&self, time: f32) -> [u8; 3] {
        match self.config.color_mode {
            ColorMode::StaticSingle => self.active_palettes.static_single_colors.first().copied().unwrap_or([255, 255, 255]),
            ColorMode::StaticMultiple => self.active_palettes.static_multi_colors.first().copied().unwrap_or([255, 255, 255]),
            ColorMode::Gradient => color::sample_gradient(&self.active_palettes.gradient_stops, 0.5),
            ColorMode::Animated => { let c = self.sample_color_changing(0.5, 0.5, time); [c.r(), c.g(), c.b()] }
        }
    }

    #[inline]
    fn apply_vis(&self, c: egui::Color32) -> egui::Color32 {
        let a = (c.a() as u32 * self.config.visualizer_opacity as u32 / 255) as u8;
        egui::Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a)
    }

    fn get_color_at(&self, i: usize, total_bars: usize, x: f32, y: f32, t: f32, rect: egui::Rect, time: f32) -> egui::Color32 {
        let c = match self.config.color_mode {
            ColorMode::StaticSingle => {
                let c = self.active_palettes.static_single_colors.first().copied().unwrap_or([255, 255, 255]);
                egui::Color32::from_rgb(c[0], c[1], c[2])
            }
            ColorMode::StaticMultiple => {
                let colors = &self.active_palettes.static_multi_colors;
                if colors.is_empty() { return self.apply_vis(egui::Color32::WHITE); }
                let num_colors = colors.len();
                if self.config.static_mode == StaticDistributionMode::Vertical {
                    let blocks = self.config.vertical_color_style == VerticalColorStyle::Blocks;
                    return self.apply_vis(static_multi_vertical_color(colors, t, blocks));
                }
                let idx = match self.config.static_mode {
                    StaticDistributionMode::Blocks => {
                        let mut chunk = (i as f32 / total_bars as f32 * num_colors as f32) as usize;
                        if chunk >= num_colors { chunk = num_colors - 1; }
                        chunk
                    }
                    StaticDistributionMode::Sequential => {
                        let bars_per_color = self.config.sequential_bars_per_color.max(1);
                        (i / bars_per_color) % num_colors
                    }
                    StaticDistributionMode::Random => random_color_index(i, num_colors),
                    StaticDistributionMode::Vertical => 0,
                };
                let c = colors.get(idx).copied().unwrap_or([255, 255, 255]);
                egui::Color32::from_rgb(c[0], c[1], c[2])
            }
            ColorMode::Gradient => {
                let angle_rad = self.config.gradient_angle.to_radians();
                let dir = egui::vec2(angle_rad.cos(), angle_rad.sin());
                let corners = [rect.left_top(), rect.right_top(), rect.left_bottom(), rect.right_bottom()];
                let mut min_proj = f32::MAX;
                let mut max_proj = f32::MIN;
                for c in corners {
                    let p = c.x * dir.x + c.y * dir.y;
                    min_proj = min_proj.min(p);
                    max_proj = max_proj.max(p);
                }
                let bar_proj = x * dir.x + y * dir.y;
                let t_grad = if (max_proj - min_proj).abs() > 0.0001 {
                    ((bar_proj - min_proj) / (max_proj - min_proj)).clamp(0.0, 1.0)
                } else { 0.0 };
                let c = color::sample_gradient(&self.active_palettes.gradient_stops, t_grad);
                egui::Color32::from_rgb(c[0], c[1], c[2])
            }
            ColorMode::Animated => {
                let norm_x = ((x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                let norm_y = ((y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0);
                self.sample_color_changing(norm_x, norm_y, time)
            }
        };
        self.apply_vis(c)
    }

    fn sample_color_changing(&self, x: f32, y: f32, time: f32) -> egui::Color32 {
        let colors = &self.active_palettes.animated_colors;
        if colors.is_empty() { return egui::Color32::WHITE; }
        let m = &self.config.color_change_motion;
        let speed_hz = 0.05 + (self.config.color_change_speed * 1.95);
        let effective_speed = speed_hz * m.speed;
        let base_t = time * effective_speed;
        let s = self.evaluate_motion(m.shape, x, y, time, base_t);
        let ca = color::cyclic_color(colors, s.phase_a);
        let cb = color::cyclic_color(colors, s.phase_b);
        let t = s.blend_t.clamp(0.0, 1.0);
        let k = s.intensity.clamp(0.0, 1.0);
        let pr = ca[0] as f32 * (1.0 - t) + cb[0] as f32 * t;
        let pg = ca[1] as f32 * (1.0 - t) + cb[1] as f32 * t;
        let pb = ca[2] as f32 * (1.0 - t) + cb[2] as f32 * t;

        let off = m.off_color;
        let off_a = m.off_color_opacity as f32 / 255.0;
        let r = off[0] as f32 * (1.0 - k) + pr * k;
        let g = off[1] as f32 * (1.0 - k) + pg * k;
        let b = off[2] as f32 * (1.0 - k) + pb * k;
        let a = off_a + (1.0 - off_a) * k;

        egui::Color32::from_rgba_unmultiplied(r as u8, g as u8, b as u8, (a * 255.0) as u8)
    }

    fn evaluate_motion(&self, shape: MotionShape, x: f32, y: f32, time: f32, base_t: f32) -> MotionSample {
        let m = &self.config.color_change_motion;
        let hard = m.blend == BlendMode::Hard;
        let freq = m.wave_frequency;
        match shape {
            MotionShape::Uniform => MotionSample { phase_a: base_t, intensity: 1.0, ..Default::default() },
            MotionShape::HorizontalWipe => {
                let boundary = tri01(base_t * 0.5);
                let t = wipe_blend(x, boundary, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::VerticalWipe => {
                let boundary = tri01(base_t * 0.5);
                let t = wipe_blend(y, boundary, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::DiagonalWipe => {
                let boundary = tri01(base_t * 0.5);
                let diag = (x + y) * 0.5;
                let t = wipe_blend(diag, boundary, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::RadialWipe => {
                let boundary = tri01(base_t * 0.5);
                let dx = x - 0.5;
                let dy = y - 0.5;
                let r = (dx * dx + dy * dy).sqrt() * 2.0;
                let t = wipe_blend(r, boundary, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::Spiral => {
                let dx = x - 0.5;
                let dy = y - 0.5;
                let r = (dx * dx + dy * dy).sqrt() * 2.0;
                let angle = dy.atan2(dx) / TAU + 0.5;
                let t = (angle + r * 2.0 + base_t).fract();
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::WaveInterference => {
                let wave1 = (x * freq * TAU + base_t * TAU).sin();
                let wave2 = (y * freq * TAU + base_t * TAU * 0.7).sin();
                let wave = (wave1 + wave2) * 0.25 + 0.5;
                MotionSample { phase_a: base_t + wave * 0.5, intensity: 1.0, ..Default::default() }
            }
            MotionShape::Ripple => {
                let dx = x - 0.5;
                let dy = y - 0.5;
                let r = (dx * dx + dy * dy).sqrt() * 2.0;
                let wave = (r * 8.0 - base_t * TAU).sin() * 0.5 + 0.5;
                MotionSample { phase_a: base_t + wave * 0.5, intensity: 1.0, ..Default::default() }
            }
            MotionShape::SineWave => { let w = (x * freq * TAU + base_t * TAU).sin() * 0.5 + 0.5; MotionSample { phase_a: base_t + w * 0.5, intensity: 1.0, ..Default::default() } }
            MotionShape::TriangleWave => { let w = tri01(x * freq + base_t); MotionSample { phase_a: base_t + w * 0.5, intensity: 1.0, ..Default::default() } }
            MotionShape::SquareWave => { let w = ((x * freq + base_t).fract() < 0.5) as u8 as f32; MotionSample { phase_a: base_t + w * 0.5, intensity: 1.0, ..Default::default() } }
            MotionShape::SawtoothWave => { let w = (x * freq + base_t).fract(); MotionSample { phase_a: base_t + w * 0.5, intensity: 1.0, ..Default::default() } }
            MotionShape::Strobe => {
                let on = ((base_t * 4.0).fract() < 0.5) as u8 as f32;
                let k = if hard { on } else { smoothstep(0.35, 0.65, on + base_t.fract() * 0.3) };
                MotionSample { phase_a: base_t, intensity: k, ..Default::default() }
            }
            MotionShape::Pulse => { let k = (base_t * TAU).sin() * 0.5 + 0.5; MotionSample { phase_a: base_t, intensity: k, ..Default::default() } }
            MotionShape::Heartbeat => {
                let beat = base_t.fract();
                let k = if hard {
                    if beat < 0.10 { 1.0 } else if beat < 0.18 { 0.0 } else if beat < 0.26 { 0.7 } else { 0.0 }
                } else {
                    let a = (-((beat - 0.08) / 0.04).powi(2)).exp();
                    let b = (-((beat - 0.22) / 0.04).powi(2)).exp() * 0.7;
                    (a + b).clamp(0.0, 1.0)
                };
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: 0.0, intensity: k }
            }
            MotionShape::Stutter => {
                let step = (time * 10.0) as u32;
                let pos = (x * 200.0) as u32;
                let hash = hash_u32(step.wrapping_mul(37).wrapping_add(pos));
                let event = (hash % 100) as f32 / 100.0 < m.stutter_intensity;
                let offset = if event { 0.5 } else { 0.0 };
                MotionSample { phase_a: base_t + offset, intensity: 1.0, ..Default::default() }
            }
            MotionShape::Comet => {
                let head = base_t.fract();
                let raw = (x - head).abs();
                let dist = raw.min(1.0 - raw);
                let k = (1.0 - dist / m.trail_length.max(0.01)).clamp(0.0, 1.0);
                let k = if hard { (k > 0.5) as u8 as f32 } else { k };
                MotionSample { phase_a: base_t, intensity: k, ..Default::default() }
            }
            MotionShape::Chase => {
                let pos = (x - base_t * 0.5).rem_euclid(1.0);
                let k = if hard { (pos < 0.15) as u8 as f32 } else { smoothstep(0.05, 0.15, 0.15 - (pos - 0.075).abs()) };
                MotionSample { phase_a: base_t, intensity: k, ..Default::default() }
            }
            MotionShape::BouncingBall => {
                let head = tri01(base_t * 0.5);
                let dist = (x - head).abs();
                let k = (1.0 - dist / m.trail_length.max(0.01)).clamp(0.0, 1.0);
                let k = if hard { (k > 0.4) as u8 as f32 } else { k };
                MotionSample { phase_a: base_t, intensity: k, ..Default::default() }
            }
            MotionShape::Sparkle => {
                let step = (time * 12.0) as u32;
                let pos = (x * 300.0) as u32;
                let hash = hash_u32(step.wrapping_mul(13).wrapping_add(pos));
                let p = (hash % 100) as f32 / 100.0;
                let on = p < 0.04 + m.stutter_intensity * 0.1;
                MotionSample { phase_a: base_t + p, intensity: if on { 1.0 } else { 0.0 }, ..Default::default() }
            }
            MotionShape::Sinelon => {
                let head = tri01(base_t * 0.5);
                let dist = (x - head).abs();
                let k = (1.0 - dist / m.trail_length.max(0.01)).clamp(0.0, 1.0);
                MotionSample { phase_a: base_t, intensity: k.powf(2.0), ..Default::default() }
            }
            MotionShape::FireSimulation => {
                let n = (base_t * 1.0 + x * 5.0).sin() + (base_t * 1.3 + x * 11.0).sin() * 0.5 + (base_t * 2.1 + x * 23.0).sin() * 0.25;
                MotionSample { phase_a: base_t + n * 0.05, intensity: (n * 0.25 + 0.75).clamp(0.0, 1.0), ..Default::default() }
            }
            MotionShape::Plasma => {
                let p = (x * 6.0 + base_t * 2.0).sin() + (y * 6.0 - base_t * 1.5).sin() + ((x + y) * 4.0 + base_t).sin();
                MotionSample { phase_a: base_t + p * 0.08, intensity: 1.0, ..Default::default() }
            }
            MotionShape::RainbowCycle => MotionSample { phase_a: base_t + x, intensity: 1.0, ..Default::default() },
            MotionShape::Confetti => {
                let step = (time * 10.0) as u32;
                let pos = (x * 400.0) as u32;
                let hash = hash_u32(step.wrapping_mul(97).wrapping_add(pos));
                let p = (hash % 100) as f32 / 100.0;
                let on = p < m.stutter_intensity * 0.15;
                MotionSample { phase_a: (hash % 1000) as f32 / 1000.0 + base_t * 0.1, intensity: if on { 1.0 } else { 0.0 }, ..Default::default() }
            }
            MotionShape::Breathing => {
                let k = (base_t * TAU * 0.5).sin() * 0.5 + 0.5;
                MotionSample { phase_a: base_t, intensity: k, ..Default::default() }
            }
            MotionShape::FlipFlop => {
                let k = if (base_t * 2.0).fract() < 0.5 { 1.0 } else { 0.0 };
                MotionSample { phase_a: base_t, intensity: k, ..Default::default() }
            }
            MotionShape::Kaleidoscope => {
                let wave = ((x * freq * TAU + base_t * TAU).sin() * (y * freq * TAU + base_t * TAU).cos()).abs();
                MotionSample { phase_a: base_t + wave * 0.5, intensity: 1.0, ..Default::default() }
            }
            MotionShape::DigitalRain => {
                let col = (x * 50.0) as u32;
                let hash = hash_u32(col);
                let speed = 1.0 + (hash % 100) as f32 / 50.0;
                let y_offset = (hash % 1000) as f32 / 1000.0;
                let pos = (y + base_t * speed + y_offset).fract();
                let k = if pos < 0.2 { 1.0 - pos / 0.2 } else { 0.0 };
                MotionSample { phase_a: base_t + col as f32 * 0.1, intensity: k, ..Default::default() }
            }
            MotionShape::MeteorShower => {
                let head = base_t.fract() * 2.0;
                let dist = ((x * 2.0 - head) % 1.0).abs();
                let k = (1.0 - dist / 0.15).clamp(0.0, 1.0);
                MotionSample { phase_a: base_t, intensity: k, ..Default::default() }
            }
        }
    }

    // ── Track info rendering ────────────────────────────────

    fn draw_styled_text(&self, ctx: &egui::Context, painter: &egui::Painter, text: &str, pos: egui::Pos2, font_id: &egui::FontId, align: egui::Align2, color: egui::Color32, cfg: &TrackInfoConfig) {
        if text.is_empty() { return; }
        let ls = cfg.letter_spacing;
        if cfg.shadow_enabled {
            let base_alpha = (cfg.opacity as f32 * 0.6).max(1.0);
            let offsets: Vec<(f32, f32)> = if cfg.shadow_blur < 1.0 { vec![(0.0, 0.0)] } else {
                let mut v = vec![(0.0, 0.0)];
                let n = 12;
                for i in 0..n { let a = i as f32 / n as f32 * TAU; v.push((a.cos() * cfg.shadow_blur, a.sin() * cfg.shadow_blur)); }
                v
            };
            let per_alpha = (base_alpha / (offsets.len() as f32).sqrt()).clamp(1.0, 255.0) as u8;
            let s_color = egui::Color32::from_rgba_unmultiplied(cfg.shadow_color[0], cfg.shadow_color[1], cfg.shadow_color[2], per_alpha);
            for (dx, dy) in offsets {
                let sp = pos + egui::vec2(cfg.shadow_offset_x + dx, cfg.shadow_offset_y + dy);
                draw_spaced(ctx, painter, text, sp, font_id, align, s_color, ls);
            }
        }
        if cfg.outline_enabled && cfg.outline_thickness > 0.1 {
            let o_color = egui::Color32::from_rgba_unmultiplied(cfg.outline_color[0], cfg.outline_color[1], cfg.outline_color[2], cfg.opacity);
            let t = cfg.outline_thickness;
            let n = 12;
            for i in 0..n {
                let a = i as f32 / n as f32 * TAU;
                let op = pos + egui::vec2(a.cos() * t, a.sin() * t);
                draw_spaced(ctx, painter, text, op, font_id, align, o_color, ls);
            }
        }
        draw_spaced(ctx, painter, text, pos, font_id, align, color, ls);
    }

    fn draw_track_info(&self, painter: &egui::Painter, ctx: &egui::Context, rect: egui::Rect) {
        let Some(info) = &self.track_info else { return };
        let cfg = &self.config.track_info;
        let time = ctx.input(|i| i.time) as f32;
        let margin = 40.0;
        let art_size = cfg.cover_size;
        let text_gap = 14.0;
        let base_rgb = match cfg.color_mode {
            TrackTextColorMode::Static => cfg.static_color,
            TrackTextColorMode::Visualizer => self.get_visualizer_text_color(time),
        };
        let text_color = egui::Color32::from_rgba_unmultiplied(base_rgb[0], base_rgb[1], base_rgb[2], cfg.opacity);
        let family = if cfg.font_path.is_some() { custom_font_family() } else { egui::FontFamily::Proportional };
        let artist_font = egui::FontId::new(cfg.artist_font_size, family.clone());
        let title_font = egui::FontId::new(cfg.title_font_size, family);
        let align = match cfg.alignment {
            TrackTextAlign::Left => egui::Align2::LEFT_TOP,
            TrackTextAlign::Center => egui::Align2::CENTER_TOP,
            TrackTextAlign::Right => egui::Align2::RIGHT_TOP,
        };
        let cover_visible = cfg.show_cover_art && self.cover_texture.is_some();
        if cover_visible {
            if let Some(tex) = &self.cover_texture {
                let art_x = rect.right() - margin - art_size + cfg.cover_x_offset;
                let art_y = rect.top() + margin + cfg.cover_y_offset;
                let art_rect = egui::Rect::from_min_size(egui::pos2(art_x, art_y), egui::vec2(art_size, art_size));
                let tint = egui::Color32::from_white_alpha(cfg.cover_opacity);
                painter.image(tex.id(), art_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), tint);
            }
        }
        let mut text_y = rect.top() + margin + cfg.text_y_offset;
        if cover_visible { text_y += art_size + text_gap; }
        let text_x = rect.right() - margin + cfg.text_x_offset;
        let mut current_y = text_y;
        if cfg.show_artist && !info.artist.is_empty() {
            self.draw_styled_text(ctx, painter, &info.artist, egui::pos2(text_x, current_y), &artist_font, align, text_color, cfg);
            current_y += cfg.artist_font_size * cfg.line_height;
        }
        if cfg.show_title && !info.title.is_empty() {
            self.draw_styled_text(ctx, painter, &info.title, egui::pos2(text_x, current_y), &title_font, align, text_color, cfg);
        }
    }
}

fn draw_spaced(ctx: &egui::Context, painter: &egui::Painter, text: &str, pos: egui::Pos2, font_id: &egui::FontId, align: egui::Align2, color: egui::Color32, letter_spacing: f32) {
    if text.is_empty() { return; }
    if letter_spacing.abs() < 0.01 { painter.text(pos, align, text, font_id.clone(), color); return; }
    let chars: Vec<char> = text.chars().collect();
    let widths: Vec<f32> = chars.iter().map(|&ch| ctx.fonts(|f| f.glyph_width(font_id, ch))).collect();
    let total_width: f32 = widths.iter().sum::<f32>() + letter_spacing * (chars.len().saturating_sub(1)) as f32;
    let start_x = match align {
        egui::Align2::LEFT_TOP | egui::Align2::LEFT_CENTER | egui::Align2::LEFT_BOTTOM => pos.x,
        egui::Align2::RIGHT_TOP | egui::Align2::RIGHT_CENTER | egui::Align2::RIGHT_BOTTOM => pos.x - total_width,
        egui::Align2::CENTER_TOP | egui::Align2::CENTER_CENTER | egui::Align2::CENTER_BOTTOM => pos.x - total_width / 2.0,
    };
    let mut x = start_x;
    for (ch, w) in chars.iter().zip(widths.iter()) {
        painter.text(egui::pos2(x, pos.y), egui::Align2::LEFT_TOP, ch.to_string(), font_id.clone(), color);
        x += w + letter_spacing;
    }
}

// ═══════════════════════════════════════════════════════════════
// Visualizer drawing
// ═══════════════════════════════════════════════════════════════

impl VisualizerApp {
    fn draw_bar_outline(&self, painter: &egui::Painter, bar_rect: egui::Rect) {
        if !self.config.visualizer_outline_enabled || self.config.visualizer_outline_thickness <= 0.0 { return; }
        let c = self.config.visualizer_outline_color;
        let color = self.apply_vis(egui::Color32::from_rgb(c[0], c[1], c[2]));
        painter.rect_stroke(bar_rect, self.config.bar_radius,
                            egui::Stroke::new(self.config.visualizer_outline_thickness, color));
    }

    fn draw_static_multi_vertical(&self, painter: &egui::Painter, rect: egui::Rect) {
        let base_num_bars = self.current_bars.len();
        if base_num_bars == 0 { return; }
        let visual_num_bars = if self.config.frequency_direction == FrequencyDirection::Mirrored {
            base_num_bars * 2
        } else { base_num_bars };
        let bar_width = self.config.bar_width;
        let gap = self.config.bar_gap;
        let stride = bar_width + gap;
        let total_size = visual_num_bars as f32 * stride - gap;
        let max_h = self.config.max_bar_height;
        let radius = self.config.bar_radius;
        let freq_dir = self.config.frequency_direction;
        let dir = self.config.visualizer_direction;
        let is_blocks = self.config.vertical_color_style == VerticalColorStyle::Blocks;
        let is_smooth = self.config.bar_interpolation == BarInterpolation::Smooth;
        let colors = &self.active_palettes.static_multi_colors;
        if colors.is_empty() { return; }
        let n_colors = colors.len();
        let n_segments: usize = if is_blocks { n_colors } else { 32 };

        let (start_pos, anchor, rect_len) = match dir {
            VisualizerDirection::Up => (
                rect.center().x - total_size / 2.0 + self.config.x_offset,
                rect.bottom() + self.config.y_offset,
                rect.height() * self.config.height_scale),
            VisualizerDirection::Down => (
                rect.center().x - total_size / 2.0 + self.config.x_offset,
                rect.top() + self.config.y_offset,
                rect.height() * self.config.height_scale),
            VisualizerDirection::Right => (
                rect.center().y - total_size / 2.0 + self.config.y_offset,
                rect.left() + self.config.x_offset,
                rect.width() * self.config.height_scale),
            VisualizerDirection::Left => (
                rect.center().y - total_size / 2.0 + self.config.y_offset,
                rect.right() + self.config.x_offset,
                rect.width() * self.config.height_scale),
        };

        if !is_smooth {
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let len = (self.current_bars[idx] * rect_len).min(if max_h > 0.0 { max_h } else { rect_len });
                if len <= 0.5 { continue; }
                let bar_rect = match dir {
                    VisualizerDirection::Up => {
                        let x = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(x, anchor - len), egui::pos2(x + bar_width, anchor))
                    }
                    VisualizerDirection::Down => {
                        let x = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(x, anchor), egui::pos2(x + bar_width, anchor + len))
                    }
                    VisualizerDirection::Right => {
                        let y = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(anchor, y), egui::pos2(anchor + len, y + bar_width))
                    }
                    VisualizerDirection::Left => {
                        let y = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(anchor - len, y), egui::pos2(anchor, y + bar_width))
                    }
                };
                for k in 0..n_segments {
                    let t0 = k as f32 / n_segments as f32;
                    let t1 = (k + 1) as f32 / n_segments as f32;
                    let t_mid = (t0 + t1) * 0.5;
                    let seg_rect = segment_rect(bar_rect, dir, t0, t1);
                    let r = segment_rounding(dir, k == 0, k == n_segments - 1, radius);
                    let c = self.apply_vis(static_multi_vertical_color(colors, t_mid, is_blocks));
                    painter.rect_filled(seg_rect, r, c);
                }
                self.draw_bar_outline(painter, bar_rect);
                if self.config.audio.peak_hold_ms > 0.0 {
                    let mut peak_len = self.current_peaks[idx] * rect_len;
                    if max_h > 0.0 { peak_len = peak_len.min(max_h); }
                    if peak_len > 0.001 {
                        let center = start_pos + i as f32 * stride + bar_width / 2.0;
                        let t = (peak_len / rect_len).clamp(0.0, 1.0);
                        let c = self.apply_vis(static_multi_vertical_color(colors, t, is_blocks));
                        draw_peak_hold(painter, dir, anchor, center, peak_len, bar_width, c, false);
                    }
                }
            }
            return;
        }

        let mut bars: Vec<(f32, f32)> = Vec::with_capacity(visual_num_bars);
        for i in 0..visual_num_bars {
            let idx = frequency_index(i, base_num_bars, freq_dir);
            let len = (self.current_bars[idx] * rect_len).min(if max_h > 0.0 { max_h } else { rect_len });
            let center = start_pos + i as f32 * stride + bar_width / 2.0;
            bars.push((center, len));
        }
        let extended: Vec<(f32, f32)> = {
            let mut v = Vec::with_capacity(bars.len() + 2);
            v.push((start_pos, bars[0].1));
            v.extend(bars.iter().copied());
            v.push((start_pos + total_size, bars.last().unwrap().1));
            v
        };
        let steps = 4usize;
        let mut mesh = egui::Mesh::default();
        macro_rules! push_quad {
            ($p1:expr, $c1:expr, $p2:expr, $c2:expr, $p3:expr, $c3:expr, $p4:expr, $c4:expr) => {{
                let base = mesh.vertices.len() as u32;
                mesh.vertices.push(egui::epaint::Vertex { pos: $p1, uv: egui::Pos2::ZERO, color: $c1 });
                mesh.vertices.push(egui::epaint::Vertex { pos: $p2, uv: egui::Pos2::ZERO, color: $c2 });
                mesh.vertices.push(egui::epaint::Vertex { pos: $p3, uv: egui::Pos2::ZERO, color: $c3 });
                mesh.vertices.push(egui::epaint::Vertex { pos: $p4, uv: egui::Pos2::ZERO, color: $c4 });
                mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }};
        }
        for i in 0..extended.len() - 1 {
            let (perp_i, len_i) = extended[i];
            let (perp_j, len_j) = extended[i + 1];
            for s in 0..steps {
                let t1 = s as f32 / steps as f32;
                let t2 = (s + 1) as f32 / steps as f32;
                let ts1 = smoothstep(0.0, 1.0, t1);
                let ts2 = smoothstep(0.0, 1.0, t2);
                let perp1 = perp_i + (perp_j - perp_i) * t1;
                let perp2 = perp_i + (perp_j - perp_i) * t2;
                let len1 = len_i + (len_j - len_i) * ts1;
                let len2 = len_i + (len_j - len_i) * ts2;
                for k in 0..n_segments {
                    let t_bot = k as f32 / n_segments as f32;
                    let t_top = (k + 1) as f32 / n_segments as f32;
                    let t_mid = (t_bot + t_top) * 0.5;
                    let c = self.apply_vis(static_multi_vertical_color(colors, t_mid, is_blocks));
                    let p1 = bar_vertex(dir, perp1, anchor, len1, t_top);
                    let p2 = bar_vertex(dir, perp2, anchor, len2, t_top);
                    let p3 = bar_vertex(dir, perp2, anchor, len2, t_bot);
                    let p4 = bar_vertex(dir, perp1, anchor, len1, t_bot);
                    push_quad!(p1, c, p2, c, p3, c, p4, c);
                }
            }
        }
        painter.add(egui::Shape::mesh(mesh));
        if self.config.visualizer_outline_enabled && self.config.visualizer_outline_thickness > 0.0 {
            for i in 0..visual_num_bars {
                let (center, len) = bars[i];
                let half = bar_width / 2.0;
                let bar_rect = match dir {
                    VisualizerDirection::Up => egui::Rect::from_min_max(egui::pos2(center - half, anchor - len), egui::pos2(center + half, anchor)),
                    VisualizerDirection::Down => egui::Rect::from_min_max(egui::pos2(center - half, anchor), egui::pos2(center + half, anchor + len)),
                    VisualizerDirection::Right => egui::Rect::from_min_max(egui::pos2(anchor, center - half), egui::pos2(anchor + len, center + half)),
                    VisualizerDirection::Left => egui::Rect::from_min_max(egui::pos2(anchor - len, center - half), egui::pos2(anchor, center + half)),
                };
                self.draw_bar_outline(painter, bar_rect);
            }
        }
        if self.config.audio.peak_hold_ms > 0.0 {
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let mut peak_len = self.current_peaks[idx] * rect_len;
                if max_h > 0.0 { peak_len = peak_len.min(max_h); }
                if peak_len > 0.001 {
                    let center = start_pos + i as f32 * stride + bar_width / 2.0;
                    let t = (peak_len / rect_len).clamp(0.0, 1.0);
                    let c = self.apply_vis(static_multi_vertical_color(colors, t, is_blocks));
                    draw_peak_hold(painter, dir, anchor, center, peak_len, bar_width, c, true);
                }
            }
        }
    }

    fn draw_animated_vertical(&self, painter: &egui::Painter, rect: egui::Rect, time: f32) {
        let base_num_bars = self.current_bars.len();
        if base_num_bars == 0 { return; }
        let visual_num_bars = if self.config.frequency_direction == FrequencyDirection::Mirrored {
            base_num_bars * 2
        } else { base_num_bars };
        let bar_width = self.config.bar_width;
        let gap = self.config.bar_gap;
        let stride = bar_width + gap;
        let total_size = visual_num_bars as f32 * stride - gap;
        let max_h = self.config.max_bar_height;
        let radius = self.config.bar_radius;
        let freq_dir = self.config.frequency_direction;
        let dir = self.config.visualizer_direction;
        let is_blocks = self.config.vertical_color_style == VerticalColorStyle::Blocks;
        let is_smooth = self.config.bar_interpolation == BarInterpolation::Smooth;
        let palette_len = self.active_palettes.animated_colors.len().max(1);
        let n_segments: usize = if is_blocks { palette_len.clamp(6, 16) } else { 24 };

        let (start_pos, anchor, rect_len) = match dir {
            VisualizerDirection::Up => (
                rect.center().x - total_size / 2.0 + self.config.x_offset,
                rect.bottom() + self.config.y_offset,
                rect.height() * self.config.height_scale),
            VisualizerDirection::Down => (
                rect.center().x - total_size / 2.0 + self.config.x_offset,
                rect.top() + self.config.y_offset,
                rect.height() * self.config.height_scale),
            VisualizerDirection::Right => (
                rect.center().y - total_size / 2.0 + self.config.y_offset,
                rect.left() + self.config.x_offset,
                rect.width() * self.config.height_scale),
            VisualizerDirection::Left => (
                rect.center().y - total_size / 2.0 + self.config.y_offset,
                rect.right() + self.config.x_offset,
                rect.width() * self.config.height_scale),
        };

        if !is_smooth {
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let len = (self.current_bars[idx] * rect_len).min(if max_h > 0.0 { max_h } else { rect_len });
                if len <= 0.5 { continue; }
                let bar_rect = match dir {
                    VisualizerDirection::Up => {
                        let x = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(x, anchor - len), egui::pos2(x + bar_width, anchor))
                    }
                    VisualizerDirection::Down => {
                        let x = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(x, anchor), egui::pos2(x + bar_width, anchor + len))
                    }
                    VisualizerDirection::Right => {
                        let y = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(anchor, y), egui::pos2(anchor + len, y + bar_width))
                    }
                    VisualizerDirection::Left => {
                        let y = start_pos + i as f32 * stride;
                        egui::Rect::from_min_max(egui::pos2(anchor - len, y), egui::pos2(anchor, y + bar_width))
                    }
                };
                for k in 0..n_segments {
                    let t0 = k as f32 / n_segments as f32;
                    let t1 = (k + 1) as f32 / n_segments as f32;
                    let seg_rect = segment_rect(bar_rect, dir, t0, t1);
                    let r = segment_rounding(dir, k == 0, k == n_segments - 1, radius);
                    let mid = seg_rect.center();
                    let x_norm = ((mid.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                    let y_norm = (t0 + t1) * 0.5;
                    let c = self.apply_vis(self.sample_color_changing(x_norm, y_norm, time));
                    painter.rect_filled(seg_rect, r, c);
                }
                self.draw_bar_outline(painter, bar_rect);
                if self.config.audio.peak_hold_ms > 0.0 {
                    let mut peak_len = self.current_peaks[idx] * rect_len;
                    if max_h > 0.0 { peak_len = peak_len.min(max_h); }
                    if peak_len > 0.001 {
                        let center = start_pos + i as f32 * stride + bar_width / 2.0;
                        let peak_pos = bar_vertex(dir, center, anchor, peak_len, 1.0);
                        let x_norm = ((peak_pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                        let c = self.apply_vis(self.sample_color_changing(x_norm, 1.0, time));
                        draw_peak_hold(painter, dir, anchor, center, peak_len, bar_width, c, false);
                    }
                }
            }
            return;
        }

        let mut bars: Vec<(f32, f32)> = Vec::with_capacity(visual_num_bars);
        for i in 0..visual_num_bars {
            let idx = frequency_index(i, base_num_bars, freq_dir);
            let len = (self.current_bars[idx] * rect_len).min(if max_h > 0.0 { max_h } else { rect_len });
            let center = start_pos + i as f32 * stride + bar_width / 2.0;
            bars.push((center, len));
        }
        let extended: Vec<(f32, f32)> = {
            let mut v = Vec::with_capacity(bars.len() + 2);
            v.push((start_pos, bars[0].1));
            v.extend(bars.iter().copied());
            v.push((start_pos + total_size, bars.last().unwrap().1));
            v
        };
        let steps = 4usize;
        let mut mesh = egui::Mesh::default();
        macro_rules! push_quad {
            ($p1:expr, $c1:expr, $p2:expr, $c2:expr, $p3:expr, $c3:expr, $p4:expr, $c4:expr) => {{
                let base = mesh.vertices.len() as u32;
                mesh.vertices.push(egui::epaint::Vertex { pos: $p1, uv: egui::Pos2::ZERO, color: $c1 });
                mesh.vertices.push(egui::epaint::Vertex { pos: $p2, uv: egui::Pos2::ZERO, color: $c2 });
                mesh.vertices.push(egui::epaint::Vertex { pos: $p3, uv: egui::Pos2::ZERO, color: $c3 });
                mesh.vertices.push(egui::epaint::Vertex { pos: $p4, uv: egui::Pos2::ZERO, color: $c4 });
                mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }};
        }
        for i in 0..extended.len() - 1 {
            let (perp_i, len_i) = extended[i];
            let (perp_j, len_j) = extended[i + 1];
            for s in 0..steps {
                let t1 = s as f32 / steps as f32;
                let t2 = (s + 1) as f32 / steps as f32;
                let ts1 = smoothstep(0.0, 1.0, t1);
                let ts2 = smoothstep(0.0, 1.0, t2);
                let perp1 = perp_i + (perp_j - perp_i) * t1;
                let perp2 = perp_i + (perp_j - perp_i) * t2;
                let len1 = len_i + (len_j - len_i) * ts1;
                let len2 = len_i + (len_j - len_i) * ts2;
                for k in 0..n_segments {
                    let t_bot = k as f32 / n_segments as f32;
                    let t_top = (k + 1) as f32 / n_segments as f32;
                    let p1 = bar_vertex(dir, perp1, anchor, len1, t_top);
                    let p2 = bar_vertex(dir, perp2, anchor, len2, t_top);
                    let p3 = bar_vertex(dir, perp2, anchor, len2, t_bot);
                    let p4 = bar_vertex(dir, perp1, anchor, len1, t_bot);
                    if is_blocks {
                        let mid = egui::pos2((p1.x + p2.x + p3.x + p4.x) * 0.25, (p1.y + p2.y + p3.y + p4.y) * 0.25);
                        let x_norm = ((mid.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                        let y_norm = (t_bot + t_top) * 0.5;
                        let c = self.apply_vis(self.sample_color_changing(x_norm, y_norm, time));
                        push_quad!(p1, c, p2, c, p3, c, p4, c);
                    } else {
                        let c1 = self.apply_vis(self.sample_color_changing(((p1.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0), t_top, time));
                        let c2 = self.apply_vis(self.sample_color_changing(((p2.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0), t_top, time));
                        let c3 = self.apply_vis(self.sample_color_changing(((p3.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0), t_bot, time));
                        let c4 = self.apply_vis(self.sample_color_changing(((p4.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0), t_bot, time));
                        push_quad!(p1, c1, p2, c2, p3, c3, p4, c4);
                    }
                }
            }
        }
        painter.add(egui::Shape::mesh(mesh));
        if self.config.visualizer_outline_enabled && self.config.visualizer_outline_thickness > 0.0 {
            for i in 0..visual_num_bars {
                let (center, len) = bars[i];
                let half = bar_width / 2.0;
                let bar_rect = match dir {
                    VisualizerDirection::Up => egui::Rect::from_min_max(egui::pos2(center - half, anchor - len), egui::pos2(center + half, anchor)),
                    VisualizerDirection::Down => egui::Rect::from_min_max(egui::pos2(center - half, anchor), egui::pos2(center + half, anchor + len)),
                    VisualizerDirection::Right => egui::Rect::from_min_max(egui::pos2(anchor, center - half), egui::pos2(anchor + len, center + half)),
                    VisualizerDirection::Left => egui::Rect::from_min_max(egui::pos2(anchor - len, center - half), egui::pos2(anchor, center + half)),
                };
                self.draw_bar_outline(painter, bar_rect);
            }
        }
        if self.config.audio.peak_hold_ms > 0.0 {
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let mut peak_len = self.current_peaks[idx] * rect_len;
                if max_h > 0.0 { peak_len = peak_len.min(max_h); }
                if peak_len > 0.001 {
                    let center = start_pos + i as f32 * stride + bar_width / 2.0;
                    let peak_pos = bar_vertex(dir, center, anchor, peak_len, 1.0);
                    let x_norm = ((peak_pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
                    let c = self.apply_vis(self.sample_color_changing(x_norm, 1.0, time));
                    draw_peak_hold(painter, dir, anchor, center, peak_len, bar_width, c, true);
                }
            }
        }
    }

    fn draw_bars(&self, painter: &egui::Painter, ctx: &egui::Context, rect: egui::Rect) {
        let base_num_bars = self.current_bars.len();
        if base_num_bars == 0 { return; }
        let time = ctx.input(|i| i.time) as f32;

        let is_static_multi_vertical = self.config.color_mode == ColorMode::StaticMultiple
            && self.config.static_mode == StaticDistributionMode::Vertical;
        if is_static_multi_vertical {
            self.draw_static_multi_vertical(painter, rect);
            return;
        }
        let is_animated_vertical = self.config.color_mode == ColorMode::Animated
            && self.config.color_change_motion.shape.uses_vertical_axis();
        if is_animated_vertical {
            self.draw_animated_vertical(painter, rect, time);
            return;
        }

        let visual_num_bars = if self.config.frequency_direction == FrequencyDirection::Mirrored {
            base_num_bars * 2
        } else { base_num_bars };
        let bar_width = self.config.bar_width;
        let gap = self.config.bar_gap;
        let stride = bar_width + gap;
        let total_size = visual_num_bars as f32 * stride - gap;
        let max_h = self.config.max_bar_height;
        let freq_dir = self.config.frequency_direction;
        let is_smooth = self.config.bar_interpolation == BarInterpolation::Smooth;

        let mut max_val: f32 = 0.0;
        for i in 0..base_num_bars { max_val = max_val.max(self.current_bars[i]); }
        let max_h_drawn = (max_val * rect.height() * self.config.height_scale)
            .min(if max_h > 0.0 { max_h } else { rect.height() * self.config.height_scale })
            .max(1.0);

        let gradient_rect = match self.config.visualizer_direction {
            VisualizerDirection::Up => egui::Rect::from_min_max(
                egui::pos2(rect.center().x - total_size / 2.0 + self.config.x_offset, rect.bottom() + self.config.y_offset - max_h_drawn),
                egui::pos2(rect.center().x + total_size / 2.0 + self.config.x_offset, rect.bottom() + self.config.y_offset)),
            VisualizerDirection::Down => egui::Rect::from_min_max(
                egui::pos2(rect.center().x - total_size / 2.0 + self.config.x_offset, rect.top() + self.config.y_offset),
                egui::pos2(rect.center().x + total_size / 2.0 + self.config.x_offset, rect.top() + self.config.y_offset + max_h_drawn)),
            VisualizerDirection::Right => egui::Rect::from_min_max(
                egui::pos2(rect.left() + self.config.x_offset, rect.center().y - total_size / 2.0 + self.config.y_offset),
                egui::pos2(rect.left() + self.config.x_offset + max_h_drawn, rect.center().y + total_size / 2.0 + self.config.y_offset)),
            VisualizerDirection::Left => egui::Rect::from_min_max(
                egui::pos2(rect.right() + self.config.x_offset - max_h_drawn, rect.center().y - total_size / 2.0 + self.config.y_offset),
                egui::pos2(rect.right() + self.config.x_offset, rect.center().y + total_size / 2.0 + self.config.y_offset)),
        };

        match self.config.visualizer_direction {
            VisualizerDirection::Up => {
                let start_x = rect.center().x - (total_size / 2.0) + self.config.x_offset;
                let bottom_y = rect.bottom() + self.config.y_offset;
                let rect_h = rect.height() * self.config.height_scale;

                if is_smooth {
                    let mut mesh = egui::Mesh::default();
                    let steps = 6;
                    let mut prev_center: Option<(f32, f32)> = None;
                    let mut prev_color_bot = egui::Color32::TRANSPARENT;
                    let mut prev_color_top = egui::Color32::TRANSPARENT;
                    let mut top_points = Vec::new();
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let h = (self.current_bars[idx] * rect_h).min(if max_h > 0.0 { max_h } else { rect_h });
                        let cx = start_x + i as f32 * stride + bar_width / 2.0;
                        let cy = bottom_y - h;
                        let color_bot = self.get_color_at(i, visual_num_bars, cx, bottom_y, 0.0, gradient_rect, time);
                        let color_top = self.get_color_at(i, visual_num_bars, cx, cy, 1.0, gradient_rect, time);
                        top_points.push(egui::pos2(cx, cy));
                        if let Some((pcx, pcy)) = prev_center {
                            for s in 0..steps {
                                let t1 = s as f32 / steps as f32;
                                let t2 = (s + 1) as f32 / steps as f32;
                                let ts1 = t1 * t1 * (3.0 - 2.0 * t1);
                                let ts2 = t2 * t2 * (3.0 - 2.0 * t2);
                                let x1 = pcx + (cx - pcx) * t1;
                                let y1 = pcy + (cy - pcy) * ts1;
                                let x2 = pcx + (cx - pcx) * t2;
                                let y2 = pcy + (cy - pcy) * ts2;
                                let c_bot1 = lerp_color(prev_color_bot, color_bot, t1);
                                let c_bot2 = lerp_color(prev_color_bot, color_bot, t2);
                                let c_top1 = lerp_color(prev_color_top, color_top, t1);
                                let c_top2 = lerp_color(prev_color_top, color_top, t2);
                                let base_idx = mesh.vertices.len() as u32;
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y1), uv: egui::Pos2::ZERO, color: c_top1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, bottom_y), uv: egui::Pos2::ZERO, color: c_bot1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, bottom_y), uv: egui::Pos2::ZERO, color: c_bot2 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, y2), uv: egui::Pos2::ZERO, color: c_top2 });
                                mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                            }
                        }
                        prev_center = Some((cx, cy));
                        prev_color_bot = color_bot;
                        prev_color_top = color_top;
                    }
                    if visual_num_bars > 0 {
                        let first = top_points.first().unwrap();
                        let last = top_points.last().unwrap();
                        let left_x = first.x - bar_width / 2.0;
                        let first_c_bot = self.get_color_at(0, visual_num_bars, left_x, bottom_y, 0.0, gradient_rect, time);
                        let first_c_top = self.get_color_at(0, visual_num_bars, left_x, first.y, 1.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, first.y), uv: egui::Pos2::ZERO, color: first_c_top });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, bottom_y), uv: egui::Pos2::ZERO, color: first_c_bot });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, bottom_y), uv: egui::Pos2::ZERO, color: first_c_bot });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, first.y), uv: egui::Pos2::ZERO, color: first_c_top });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                        let right_x = last.x + bar_width / 2.0;
                        let last_c_bot = self.get_color_at(visual_num_bars-1, visual_num_bars, right_x, bottom_y, 0.0, gradient_rect, time);
                        let last_c_top = self.get_color_at(visual_num_bars-1, visual_num_bars, right_x, last.y, 1.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, last.y), uv: egui::Pos2::ZERO, color: last_c_top });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, bottom_y), uv: egui::Pos2::ZERO, color: last_c_bot });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, bottom_y), uv: egui::Pos2::ZERO, color: last_c_bot });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, last.y), uv: egui::Pos2::ZERO, color: last_c_top });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                    }
                    painter.add(egui::Shape::mesh(mesh));
                    if self.config.visualizer_outline_enabled && self.config.visualizer_outline_thickness > 0.0 {
                        let mut pts = Vec::with_capacity(top_points.len() + 3);
                        pts.extend_from_slice(&top_points);
                        if let Some(last) = top_points.last() { pts.push(egui::pos2(last.x + bar_width / 2.0, bottom_y)); }
                        if let Some(first) = top_points.first() { pts.push(egui::pos2(first.x - bar_width / 2.0, bottom_y)); pts.push(*first); }
                        let c = self.config.visualizer_outline_color;
                        let stroke = egui::Stroke::new(self.config.visualizer_outline_thickness, self.apply_vis(egui::Color32::from_rgb(c[0], c[1], c[2])));
                        painter.add(egui::Shape::line(pts, stroke));
                    }
                    if self.config.audio.peak_hold_ms > 0.0 {
                        for i in 0..visual_num_bars {
                            let idx = frequency_index(i, base_num_bars, freq_dir);
                            let mut peak_h = self.current_peaks[idx] * rect_h;
                            if max_h > 0.0 { peak_h = peak_h.min(max_h); }
                            if peak_h > 0.001 {
                                let x_center = start_x + i as f32 * stride + bar_width / 2.0;
                                let cy = bottom_y - peak_h;
                                let color = self.get_color_at(i, visual_num_bars, x_center, cy, 1.0, gradient_rect, time);
                                draw_peak_hold(painter, VisualizerDirection::Up, bottom_y, x_center, peak_h, bar_width, color, true);
                            }
                        }
                    }
                    return;
                }

                let mut mesh = egui::Mesh::default();
                for i in 0..visual_num_bars {
                    let idx = frequency_index(i, base_num_bars, freq_dir);
                    let h = (self.current_bars[idx] * rect_h).min(if max_h > 0.0 { max_h } else { rect_h });
                    let x0 = start_x + i as f32 * stride;
                    let x1 = x0 + bar_width;
                    let y_top = bottom_y - h;
                    let c_top = self.get_color_at(i, visual_num_bars, x0 + bar_width / 2.0, y_top, 1.0, gradient_rect, time);
                    let c_bot = self.get_color_at(i, visual_num_bars, x0 + bar_width / 2.0, bottom_y, 0.0, gradient_rect, time);
                    let base_idx = mesh.vertices.len() as u32;
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x0, y_top), uv: egui::Pos2::ZERO, color: c_top });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x0, bottom_y), uv: egui::Pos2::ZERO, color: c_bot });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, bottom_y), uv: egui::Pos2::ZERO, color: c_bot });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y_top), uv: egui::Pos2::ZERO, color: c_top });
                    mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                }
                painter.add(egui::Shape::mesh(mesh));
                if self.config.visualizer_outline_enabled {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let h = (self.current_bars[idx] * rect_h).min(if max_h > 0.0 { max_h } else { rect_h });
                        let x = start_x + i as f32 * stride;
                        let bar_rect = egui::Rect::from_min_max(egui::pos2(x, bottom_y - h), egui::pos2(x + bar_width, bottom_y));
                        self.draw_bar_outline(painter, bar_rect);
                    }
                }
                if self.config.audio.peak_hold_ms > 0.0 {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let mut peak_h = self.current_peaks[idx] * rect_h;
                        if max_h > 0.0 { peak_h = peak_h.min(max_h); }
                        if peak_h > 0.001 {
                            let x_center = start_x + i as f32 * stride + bar_width / 2.0;
                            let color = self.get_color_at(i, visual_num_bars, x_center, bottom_y - peak_h, 1.0, gradient_rect, time);
                            draw_peak_hold(painter, VisualizerDirection::Up, bottom_y, x_center, peak_h, bar_width, color, false);
                        }
                    }
                }
            }
            VisualizerDirection::Down => {
                let start_x = rect.center().x - (total_size / 2.0) + self.config.x_offset;
                let top_y = rect.top() + self.config.y_offset;
                let rect_h = rect.height() * self.config.height_scale;
                if is_smooth {
                    let mut mesh = egui::Mesh::default();
                    let steps = 6;
                    let mut prev_center: Option<(f32, f32)> = None;
                    let mut prev_color_bot = egui::Color32::TRANSPARENT;
                    let mut prev_color_top = egui::Color32::TRANSPARENT;
                    let mut top_points = Vec::new();
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let h = (self.current_bars[idx] * rect_h).min(if max_h > 0.0 { max_h } else { rect_h });
                        let cx = start_x + i as f32 * stride + bar_width / 2.0;
                        let cy = top_y + h;
                        let color_top = self.get_color_at(i, visual_num_bars, cx, top_y, 0.0, gradient_rect, time);
                        let color_bot = self.get_color_at(i, visual_num_bars, cx, cy, 1.0, gradient_rect, time);
                        top_points.push(egui::pos2(cx, cy));
                        if let Some((pcx, pcy)) = prev_center {
                            for s in 0..steps {
                                let t1 = s as f32 / steps as f32;
                                let t2 = (s + 1) as f32 / steps as f32;
                                let ts1 = t1 * t1 * (3.0 - 2.0 * t1);
                                let ts2 = t2 * t2 * (3.0 - 2.0 * t2);
                                let x1 = pcx + (cx - pcx) * t1;
                                let y1 = pcy + (cy - pcy) * ts1;
                                let x2 = pcx + (cx - pcx) * t2;
                                let y2 = pcy + (cy - pcy) * ts2;
                                let c_top1 = lerp_color(prev_color_top, color_top, t1);
                                let c_top2 = lerp_color(prev_color_top, color_top, t2);
                                let c_bot1 = lerp_color(prev_color_bot, color_bot, t1);
                                let c_bot2 = lerp_color(prev_color_bot, color_bot, t2);
                                let base_idx = mesh.vertices.len() as u32;
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, top_y), uv: egui::Pos2::ZERO, color: c_top1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y1), uv: egui::Pos2::ZERO, color: c_bot1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, y2), uv: egui::Pos2::ZERO, color: c_bot2 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, top_y), uv: egui::Pos2::ZERO, color: c_top2 });
                                mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                            }
                        }
                        prev_center = Some((cx, cy));
                        prev_color_bot = color_bot;
                        prev_color_top = color_top;
                    }
                    if visual_num_bars > 0 {
                        let first = top_points.first().unwrap();
                        let last = top_points.last().unwrap();
                        let left_x = first.x - bar_width / 2.0;
                        let first_c_top = self.get_color_at(0, visual_num_bars, left_x, top_y, 0.0, gradient_rect, time);
                        let first_c_bot = self.get_color_at(0, visual_num_bars, left_x, first.y, 1.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, top_y), uv: egui::Pos2::ZERO, color: first_c_top });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, top_y), uv: egui::Pos2::ZERO, color: first_c_top });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, first.y), uv: egui::Pos2::ZERO, color: first_c_bot });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, first.y), uv: egui::Pos2::ZERO, color: first_c_bot });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                        let right_x = last.x + bar_width / 2.0;
                        let last_c_top = self.get_color_at(visual_num_bars-1, visual_num_bars, right_x, top_y, 0.0, gradient_rect, time);
                        let last_c_bot = self.get_color_at(visual_num_bars-1, visual_num_bars, right_x, last.y, 1.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, top_y), uv: egui::Pos2::ZERO, color: last_c_top });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, top_y), uv: egui::Pos2::ZERO, color: last_c_top });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, last.y), uv: egui::Pos2::ZERO, color: last_c_bot });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, last.y), uv: egui::Pos2::ZERO, color: last_c_bot });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                    }
                    painter.add(egui::Shape::mesh(mesh));
                    if self.config.visualizer_outline_enabled && self.config.visualizer_outline_thickness > 0.0 {
                        let mut pts = Vec::with_capacity(top_points.len() + 3);
                        pts.extend_from_slice(&top_points);
                        if let Some(last) = top_points.last() { pts.push(egui::pos2(last.x + bar_width / 2.0, top_y)); }
                        if let Some(first) = top_points.first() { pts.push(egui::pos2(first.x - bar_width / 2.0, top_y)); pts.push(*first); }
                        let c = self.config.visualizer_outline_color;
                        let stroke = egui::Stroke::new(self.config.visualizer_outline_thickness, self.apply_vis(egui::Color32::from_rgb(c[0], c[1], c[2])));
                        painter.add(egui::Shape::line(pts, stroke));
                    }
                    if self.config.audio.peak_hold_ms > 0.0 {
                        for i in 0..visual_num_bars {
                            let idx = frequency_index(i, base_num_bars, freq_dir);
                            let mut peak_h = self.current_peaks[idx] * rect_h;
                            if max_h > 0.0 { peak_h = peak_h.min(max_h); }
                            if peak_h > 0.001 {
                                let x_center = start_x + i as f32 * stride + bar_width / 2.0;
                                let cy = top_y + peak_h;
                                let color = self.get_color_at(i, visual_num_bars, x_center, cy, 1.0, gradient_rect, time);
                                draw_peak_hold(painter, VisualizerDirection::Down, top_y, x_center, peak_h, bar_width, color, true);
                            }
                        }
                    }
                    return;
                }
                let mut mesh = egui::Mesh::default();
                for i in 0..visual_num_bars {
                    let idx = frequency_index(i, base_num_bars, freq_dir);
                    let h = (self.current_bars[idx] * rect_h).min(if max_h > 0.0 { max_h } else { rect_h });
                    let x0 = start_x + i as f32 * stride;
                    let x1 = x0 + bar_width;
                    let y_bot = top_y + h;
                    let c_top = self.get_color_at(i, visual_num_bars, x0 + bar_width / 2.0, top_y, 0.0, gradient_rect, time);
                    let c_bot = self.get_color_at(i, visual_num_bars, x0 + bar_width / 2.0, y_bot, 1.0, gradient_rect, time);
                    let base_idx = mesh.vertices.len() as u32;
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x0, top_y), uv: egui::Pos2::ZERO, color: c_top });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x0, y_bot), uv: egui::Pos2::ZERO, color: c_bot });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y_bot), uv: egui::Pos2::ZERO, color: c_bot });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, top_y), uv: egui::Pos2::ZERO, color: c_top });
                    mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                }
                painter.add(egui::Shape::mesh(mesh));
                if self.config.visualizer_outline_enabled {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let h = (self.current_bars[idx] * rect_h).min(if max_h > 0.0 { max_h } else { rect_h });
                        let x = start_x + i as f32 * stride;
                        let bar_rect = egui::Rect::from_min_max(egui::pos2(x, top_y), egui::pos2(x + bar_width, top_y + h));
                        self.draw_bar_outline(painter, bar_rect);
                    }
                }
                if self.config.audio.peak_hold_ms > 0.0 {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let mut peak_h = self.current_peaks[idx] * rect_h;
                        if max_h > 0.0 { peak_h = peak_h.min(max_h); }
                        if peak_h > 0.001 {
                            let x_center = start_x + i as f32 * stride + bar_width / 2.0;
                            let color = self.get_color_at(i, visual_num_bars, x_center, top_y + peak_h, 1.0, gradient_rect, time);
                            draw_peak_hold(painter, VisualizerDirection::Down, top_y, x_center, peak_h, bar_width, color, false);
                        }
                    }
                }
            }
            VisualizerDirection::Right => {
                let left_x = rect.left() + self.config.x_offset;
                let start_y = rect.center().y - (total_size / 2.0) + self.config.y_offset;
                let rect_w = rect.width() * self.config.height_scale;
                if is_smooth {
                    let mut mesh = egui::Mesh::default();
                    let steps = 6;
                    let mut prev_center: Option<(f32, f32)> = None;
                    let mut prev_color_left = egui::Color32::TRANSPARENT;
                    let mut prev_color_right = egui::Color32::TRANSPARENT;
                    let mut top_points = Vec::new();
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let w = (self.current_bars[idx] * rect_w).min(if max_h > 0.0 { max_h } else { rect_w });
                        let cx = left_x + w;
                        let cy = start_y + i as f32 * stride + bar_width / 2.0;
                        let color_left = self.get_color_at(i, visual_num_bars, left_x, cy, 0.0, gradient_rect, time);
                        let color_right = self.get_color_at(i, visual_num_bars, cx, cy, 1.0, gradient_rect, time);
                        top_points.push(egui::pos2(cx, cy));
                        if let Some((pcx, pcy)) = prev_center {
                            for s in 0..steps {
                                let t1 = s as f32 / steps as f32;
                                let t2 = (s + 1) as f32 / steps as f32;
                                let ts1 = t1 * t1 * (3.0 - 2.0 * t1);
                                let ts2 = t2 * t2 * (3.0 - 2.0 * t2);
                                let x1 = pcx + (cx - pcx) * ts1;
                                let y1 = pcy + (cy - pcy) * t1;
                                let x2 = pcx + (cx - pcx) * ts2;
                                let y2 = pcy + (cy - pcy) * t2;
                                let c_left1 = lerp_color(prev_color_left, color_left, t1);
                                let c_left2 = lerp_color(prev_color_left, color_left, t2);
                                let c_right1 = lerp_color(prev_color_right, color_right, t1);
                                let c_right2 = lerp_color(prev_color_right, color_right, t2);
                                let base_idx = mesh.vertices.len() as u32;
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, y1), uv: egui::Pos2::ZERO, color: c_left1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y1), uv: egui::Pos2::ZERO, color: c_right1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, y2), uv: egui::Pos2::ZERO, color: c_right2 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, y2), uv: egui::Pos2::ZERO, color: c_left2 });
                                mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                            }
                        }
                        prev_center = Some((cx, cy));
                        prev_color_left = color_left;
                        prev_color_right = color_right;
                    }
                    if visual_num_bars > 0 {
                        let first = top_points.first().unwrap();
                        let last = top_points.last().unwrap();
                        let top_y_first = first.y - bar_width / 2.0;
                        let first_c_left = self.get_color_at(0, visual_num_bars, left_x, top_y_first, 0.0, gradient_rect, time);
                        let first_c_right = self.get_color_at(0, visual_num_bars, first.x, top_y_first, 1.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, top_y_first), uv: egui::Pos2::ZERO, color: first_c_left });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, top_y_first), uv: egui::Pos2::ZERO, color: first_c_right });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, first.y), uv: egui::Pos2::ZERO, color: first_c_right });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, first.y), uv: egui::Pos2::ZERO, color: first_c_left });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                        let bot_y_last = last.y + bar_width / 2.0;
                        let last_c_left = self.get_color_at(visual_num_bars-1, visual_num_bars, left_x, bot_y_last, 0.0, gradient_rect, time);
                        let last_c_right = self.get_color_at(visual_num_bars-1, visual_num_bars, last.x, bot_y_last, 1.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, last.y), uv: egui::Pos2::ZERO, color: last_c_left });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, last.y), uv: egui::Pos2::ZERO, color: last_c_right });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, bot_y_last), uv: egui::Pos2::ZERO, color: last_c_right });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, bot_y_last), uv: egui::Pos2::ZERO, color: last_c_left });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                    }
                    painter.add(egui::Shape::mesh(mesh));
                    if self.config.visualizer_outline_enabled && self.config.visualizer_outline_thickness > 0.0 {
                        let mut pts = Vec::with_capacity(top_points.len() + 3);
                        pts.extend_from_slice(&top_points);
                        if let Some(last) = top_points.last() { pts.push(egui::pos2(last.x, last.y + bar_width / 2.0)); }
                        if let Some(first) = top_points.first() { pts.push(egui::pos2(first.x, first.y - bar_width / 2.0)); pts.push(*first); }
                        let c = self.config.visualizer_outline_color;
                        let stroke = egui::Stroke::new(self.config.visualizer_outline_thickness, self.apply_vis(egui::Color32::from_rgb(c[0], c[1], c[2])));
                        painter.add(egui::Shape::line(pts, stroke));
                    }
                    if self.config.audio.peak_hold_ms > 0.0 {
                        for i in 0..visual_num_bars {
                            let idx = frequency_index(i, base_num_bars, freq_dir);
                            let mut peak_w = self.current_peaks[idx] * rect_w;
                            if max_h > 0.0 { peak_w = peak_w.min(max_h); }
                            if peak_w > 0.001 {
                                let y_center = start_y + i as f32 * stride + bar_width / 2.0;
                                let cx = left_x + peak_w;
                                let color = self.get_color_at(i, visual_num_bars, cx, y_center, 1.0, gradient_rect, time);
                                draw_peak_hold(painter, VisualizerDirection::Right, left_x, y_center, peak_w, bar_width, color, true);
                            }
                        }
                    }
                    return;
                }
                let mut mesh = egui::Mesh::default();
                for i in 0..visual_num_bars {
                    let idx = frequency_index(i, base_num_bars, freq_dir);
                    let w = (self.current_bars[idx] * rect_w).min(if max_h > 0.0 { max_h } else { rect_w });
                    let y0 = start_y + i as f32 * stride;
                    let y1 = y0 + bar_width;
                    let x1 = left_x + w;
                    let c_left = self.get_color_at(i, visual_num_bars, left_x, y0 + bar_width / 2.0, 0.0, gradient_rect, time);
                    let c_right = self.get_color_at(i, visual_num_bars, x1, y0 + bar_width / 2.0, 1.0, gradient_rect, time);
                    let base_idx = mesh.vertices.len() as u32;
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, y0), uv: egui::Pos2::ZERO, color: c_left });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y0), uv: egui::Pos2::ZERO, color: c_right });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y1), uv: egui::Pos2::ZERO, color: c_right });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(left_x, y1), uv: egui::Pos2::ZERO, color: c_left });
                    mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                }
                painter.add(egui::Shape::mesh(mesh));
                if self.config.visualizer_outline_enabled {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let w = (self.current_bars[idx] * rect_w).min(if max_h > 0.0 { max_h } else { rect_w });
                        let y = start_y + i as f32 * stride;
                        let bar_rect = egui::Rect::from_min_max(egui::pos2(left_x, y), egui::pos2(left_x + w, y + bar_width));
                        self.draw_bar_outline(painter, bar_rect);
                    }
                }
                if self.config.audio.peak_hold_ms > 0.0 {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let mut peak_w = self.current_peaks[idx] * rect_w;
                        if max_h > 0.0 { peak_w = peak_w.min(max_h); }
                        if peak_w > 0.001 {
                            let y_center = start_y + i as f32 * stride + bar_width / 2.0;
                            let color = self.get_color_at(i, visual_num_bars, left_x + peak_w, y_center, 1.0, gradient_rect, time);
                            draw_peak_hold(painter, VisualizerDirection::Right, left_x, y_center, peak_w, bar_width, color, false);
                        }
                    }
                }
            }
            VisualizerDirection::Left => {
                let right_x = rect.right() + self.config.x_offset;
                let start_y = rect.center().y - (total_size / 2.0) + self.config.y_offset;
                let rect_w = rect.width() * self.config.height_scale;
                if is_smooth {
                    let mut mesh = egui::Mesh::default();
                    let steps = 6;
                    let mut prev_center: Option<(f32, f32)> = None;
                    let mut prev_color_left = egui::Color32::TRANSPARENT;
                    let mut prev_color_right = egui::Color32::TRANSPARENT;
                    let mut top_points = Vec::new();
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let w = (self.current_bars[idx] * rect_w).min(if max_h > 0.0 { max_h } else { rect_w });
                        let cx = right_x - w;
                        let cy = start_y + i as f32 * stride + bar_width / 2.0;
                        let color_left = self.get_color_at(i, visual_num_bars, cx, cy, 1.0, gradient_rect, time);
                        let color_right = self.get_color_at(i, visual_num_bars, right_x, cy, 0.0, gradient_rect, time);
                        top_points.push(egui::pos2(cx, cy));
                        if let Some((pcx, pcy)) = prev_center {
                            for s in 0..steps {
                                let t1 = s as f32 / steps as f32;
                                let t2 = (s + 1) as f32 / steps as f32;
                                let ts1 = t1 * t1 * (3.0 - 2.0 * t1);
                                let ts2 = t2 * t2 * (3.0 - 2.0 * t2);
                                let x1 = pcx + (cx - pcx) * ts1;
                                let y1 = pcy + (cy - pcy) * t1;
                                let x2 = pcx + (cx - pcx) * ts2;
                                let y2 = pcy + (cy - pcy) * t2;
                                let c_left1 = lerp_color(prev_color_left, color_left, t1);
                                let c_left2 = lerp_color(prev_color_left, color_left, t2);
                                let c_right1 = lerp_color(prev_color_right, color_right, t1);
                                let c_right2 = lerp_color(prev_color_right, color_right, t2);
                                let base_idx = mesh.vertices.len() as u32;
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, y1), uv: egui::Pos2::ZERO, color: c_right1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, y1), uv: egui::Pos2::ZERO, color: c_left1 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, y2), uv: egui::Pos2::ZERO, color: c_left2 });
                                mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, y2), uv: egui::Pos2::ZERO, color: c_right2 });
                                mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                            }
                        }
                        prev_center = Some((cx, cy));
                        prev_color_left = color_left;
                        prev_color_right = color_right;
                    }
                    if visual_num_bars > 0 {
                        let first = top_points.first().unwrap();
                        let last = top_points.last().unwrap();
                        let top_y_first = first.y - bar_width / 2.0;
                        let first_c_left = self.get_color_at(0, visual_num_bars, first.x, top_y_first, 1.0, gradient_rect, time);
                        let first_c_right = self.get_color_at(0, visual_num_bars, right_x, top_y_first, 0.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, top_y_first), uv: egui::Pos2::ZERO, color: first_c_left });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(first.x, first.y), uv: egui::Pos2::ZERO, color: first_c_left });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, first.y), uv: egui::Pos2::ZERO, color: first_c_right });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, top_y_first), uv: egui::Pos2::ZERO, color: first_c_right });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                        let bot_y_last = last.y + bar_width / 2.0;
                        let last_c_left = self.get_color_at(visual_num_bars-1, visual_num_bars, last.x, bot_y_last, 1.0, gradient_rect, time);
                        let last_c_right = self.get_color_at(visual_num_bars-1, visual_num_bars, right_x, bot_y_last, 0.0, gradient_rect, time);
                        let b_idx = mesh.vertices.len() as u32;
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, last.y), uv: egui::Pos2::ZERO, color: last_c_left });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(last.x, bot_y_last), uv: egui::Pos2::ZERO, color: last_c_left });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, bot_y_last), uv: egui::Pos2::ZERO, color: last_c_right });
                        mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, last.y), uv: egui::Pos2::ZERO, color: last_c_right });
                        mesh.indices.extend_from_slice(&[b_idx, b_idx+1, b_idx+2, b_idx, b_idx+2, b_idx+3]);
                    }
                    painter.add(egui::Shape::mesh(mesh));
                    if self.config.visualizer_outline_enabled && self.config.visualizer_outline_thickness > 0.0 {
                        let mut pts = Vec::with_capacity(top_points.len() + 3);
                        pts.extend_from_slice(&top_points);
                        if let Some(last) = top_points.last() { pts.push(egui::pos2(last.x, last.y + bar_width / 2.0)); }
                        if let Some(first) = top_points.first() { pts.push(egui::pos2(first.x, first.y - bar_width / 2.0)); pts.push(*first); }
                        let c = self.config.visualizer_outline_color;
                        let stroke = egui::Stroke::new(self.config.visualizer_outline_thickness, self.apply_vis(egui::Color32::from_rgb(c[0], c[1], c[2])));
                        painter.add(egui::Shape::line(pts, stroke));
                    }
                    if self.config.audio.peak_hold_ms > 0.0 {
                        for i in 0..visual_num_bars {
                            let idx = frequency_index(i, base_num_bars, freq_dir);
                            let mut peak_w = self.current_peaks[idx] * rect_w;
                            if max_h > 0.0 { peak_w = peak_w.min(max_h); }
                            if peak_w > 0.001 {
                                let y_center = start_y + i as f32 * stride + bar_width / 2.0;
                                let cx = right_x - peak_w;
                                let color = self.get_color_at(i, visual_num_bars, cx, y_center, 1.0, gradient_rect, time);
                                draw_peak_hold(painter, VisualizerDirection::Left, right_x, y_center, peak_w, bar_width, color, true);
                            }
                        }
                    }
                    return;
                }
                let mut mesh = egui::Mesh::default();
                for i in 0..visual_num_bars {
                    let idx = frequency_index(i, base_num_bars, freq_dir);
                    let w = (self.current_bars[idx] * rect_w).min(if max_h > 0.0 { max_h } else { rect_w });
                    let y0 = start_y + i as f32 * stride;
                    let y1 = y0 + bar_width;
                    let x0 = right_x - w;
                    let c_left = self.get_color_at(i, visual_num_bars, x0, y0 + bar_width / 2.0, 1.0, gradient_rect, time);
                    let c_right = self.get_color_at(i, visual_num_bars, right_x, y0 + bar_width / 2.0, 0.0, gradient_rect, time);
                    let base_idx = mesh.vertices.len() as u32;
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x0, y0), uv: egui::Pos2::ZERO, color: c_left });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, y0), uv: egui::Pos2::ZERO, color: c_right });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(right_x, y1), uv: egui::Pos2::ZERO, color: c_right });
                    mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x0, y1), uv: egui::Pos2::ZERO, color: c_left });
                    mesh.indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);
                }
                painter.add(egui::Shape::mesh(mesh));
                if self.config.visualizer_outline_enabled {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let w = (self.current_bars[idx] * rect_w).min(if max_h > 0.0 { max_h } else { rect_w });
                        let y = start_y + i as f32 * stride;
                        let bar_rect = egui::Rect::from_min_max(egui::pos2(right_x - w, y), egui::pos2(right_x, y + bar_width));
                        self.draw_bar_outline(painter, bar_rect);
                    }
                }
                if self.config.audio.peak_hold_ms > 0.0 {
                    for i in 0..visual_num_bars {
                        let idx = frequency_index(i, base_num_bars, freq_dir);
                        let mut peak_w = self.current_peaks[idx] * rect_w;
                        if max_h > 0.0 { peak_w = peak_w.min(max_h); }
                        if peak_w > 0.001 {
                            let y_center = start_y + i as f32 * stride + bar_width / 2.0;
                            let color = self.get_color_at(i, visual_num_bars, right_x - peak_w, y_center, 1.0, gradient_rect, time);
                            draw_peak_hold(painter, VisualizerDirection::Left, right_x, y_center, peak_w, bar_width, color, false);
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Settings UI
// ═══════════════════════════════════════════════════════════════

impl SettingsState {
    fn draw_settings_ui(&mut self, ui: &mut egui::Ui) {
        let scale = self.config.settings_scale.clamp(0.75, 1.5);
        
        egui::SidePanel::left("settings_nav")
            .resizable(false)
            .exact_width(140.0)
            .frame(egui::Frame::none().fill(egui::Color32::BLACK).inner_margin(egui::Margin::same(8.0)))
            .show_inside(ui, |ui| {
                self.draw_sidebar(ui);
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::BLACK).inner_margin(egui::Margin::same(12.0)))
            .show_inside(ui, |ui| {
                ui.style_mut().text_styles.insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
                ui.style_mut().text_styles.insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
                ui.style_mut().text_styles.insert(egui::TextStyle::Small, egui::FontId::proportional(12.0));
                ui.style_mut().text_styles.insert(egui::TextStyle::Heading, egui::FontId::proportional(18.0));

                ui.style_mut().spacing.item_spacing *= scale;
                ui.style_mut().spacing.button_padding *= scale;
                ui.style_mut().spacing.slider_width *= scale;
                ui.style_mut().spacing.combo_width *= scale;
                
                for (_, font) in ui.style_mut().text_styles.iter_mut() {
                    font.size *= scale;
                }

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        match self.settings_tab {
                            SettingsTab::Visualizer  => self.tab_visualizer(ui),
                            SettingsTab::Colors      => self.tab_colors(ui),
                            SettingsTab::Audio       => self.tab_audio(ui),
                            SettingsTab::TrackInfo   => self.tab_track_info(ui),
                            SettingsTab::Appearance  => self.tab_appearance(ui),
                            SettingsTab::Performance => self.tab_performance(ui),
                        }
                    });
            });
    }

    fn draw_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Settings").size(16.0).strong().color(accent()));
        ui.add_space(8.0);

        ui.add(egui::TextEdit::singleline(&mut self.settings_search)
            .hint_text("Search…")
            .desired_width(f32::INFINITY));
        ui.add_space(6.0);

        let query = self.settings_search.trim().to_lowercase();
        if !query.is_empty() {
            ui.label(egui::RichText::new("Results").small().weak());
            let mut any = false;
            for (tab, name, keywords) in SEARCH_INDEX {
                let hit = name.to_lowercase().contains(&query)
                    || keywords.iter().any(|k| k.contains(&query));
                if hit {
                    any = true;
                    if ui.small_button(*name).clicked() {
                        self.settings_tab = *tab;
                    }
                }
            }
            if !any {
                ui.label(egui::RichText::new("No matches").small().weak());
            }
            ui.add_space(6.0);
            ui.separator();
        }

        ui.add_space(4.0);
        let tabs = [
            (SettingsTab::Visualizer,  "Visualizer"),
            (SettingsTab::Colors,      "Colors"),
            (SettingsTab::Audio,       "Audio"),
            (SettingsTab::TrackInfo,   "Track Info"),
            (SettingsTab::Appearance,  "Appearance"),
            (SettingsTab::Performance, "Performance"),
        ];
        for (tab, label) in tabs {
            let selected = self.settings_tab == tab;
            let mut text = egui::RichText::new(label).size(13.0);
            if selected { text = text.strong().color(accent()); }
            if ui.add_sized([ui.available_width(), 26.0],
                            egui::SelectableLabel::new(selected, text)).clicked() {
                self.settings_tab = tab;
            }
        }

        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.add_space(8.0);
            if ui.add_sized([ui.available_width(), 24.0], 
                egui::Button::new(egui::RichText::new("Close").size(13.0)).frame(false)).clicked() 
            {
                self.close_requested = true;
                self.config.save();
            }
            if ui.add_sized([ui.available_width(), 24.0], 
                egui::Button::new(egui::RichText::new("Reset All").size(13.0).color(egui::Color32::from_rgb(255, 100, 100))).frame(false)).clicked() 
            {
                self.config = AppConfig::default();
                self.audio_dirty = true;
            }
        });
    }

    // ── Visualizer tab ───────────────────────────────────────

    fn tab_visualizer(&mut self, ui: &mut egui::Ui) {
        use cfg::*;

        section_header(ui, "Direction");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.config.visualizer_direction, VisualizerDirection::Up, "Up");
            ui.radio_value(&mut self.config.visualizer_direction, VisualizerDirection::Down, "Down");
            ui.radio_value(&mut self.config.visualizer_direction, VisualizerDirection::Left, "Left");
            ui.radio_value(&mut self.config.visualizer_direction, VisualizerDirection::Right, "Right");
            reset_btn(ui, self.config.visualizer_direction != VisualizerDirection::Up,
                      || self.config.visualizer_direction = VisualizerDirection::Up);
        });
        help_text(ui, self.config.show_help_text, "Direction bars grow from.");

        section_header(ui, "Frequency Layout");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.config.frequency_direction, FrequencyDirection::LeftToRight, "Left to Right");
            ui.radio_value(&mut self.config.frequency_direction, FrequencyDirection::RightToLeft, "Right to Left");
            ui.radio_value(&mut self.config.frequency_direction, FrequencyDirection::Mirrored, "Mirrored (Symmetrical)");
            reset_btn(ui, self.config.frequency_direction != FrequencyDirection::LeftToRight,
                      || self.config.frequency_direction = FrequencyDirection::LeftToRight);
        });
        help_text(ui, self.config.show_help_text, "How the frequency spectrum is laid out across the screen.");

        section_header(ui, "Bar Shape");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.config.bar_interpolation, BarInterpolation::Step, "Step");
            ui.radio_value(&mut self.config.bar_interpolation, BarInterpolation::Smooth, "Smooth");
            reset_btn(ui, self.config.bar_interpolation != BarInterpolation::Step,
                      || self.config.bar_interpolation = BarInterpolation::Step);
        });
        help_text(ui, self.config.show_help_text, self.config.bar_interpolation.description());

        section_header(ui, "Bars");
        slider_f32(ui, &mut self.config.bar_width, 1.0..=50.0, "Bar Width", DEFAULT_BAR_WIDTH);
        slider_f32(ui, &mut self.config.bar_gap, 0.0..=20.0, "Bar Gap", DEFAULT_BAR_GAP);
        ui.horizontal(|ui| {
            let mut n = self.config.audio.num_bars as i32;
            let r = ui.add(egui::Slider::new(&mut n, 16..=256).text("Bar Count"));
            if r.secondary_clicked() { n = DEFAULT_NUM_BARS as i32; }
            if n as usize != self.config.audio.num_bars { self.config.audio.num_bars = n as usize; }
            reset_btn(ui, n as usize != DEFAULT_NUM_BARS, || self.config.audio.num_bars = DEFAULT_NUM_BARS);
        });
        slider_f32(ui, &mut self.config.max_bar_height, 0.0..=1000.0, "Max Bar Height (0 = full)", DEFAULT_MAX_BAR_HEIGHT);
        slider_f32(ui, &mut self.config.height_scale, 0.1..=1.0, "Height Scale", DEFAULT_HEIGHT_SCALE);

        section_header(ui, "Position");
        slider_f32(ui, &mut self.config.x_offset, -1000.0..=1000.0, "Horizontal Offset", DEFAULT_X_OFFSET);
        slider_f32(ui, &mut self.config.y_offset, -1000.0..=1000.0, "Vertical Offset", DEFAULT_Y_OFFSET);

        section_header(ui, "Appearance");
        slider_u8(ui, &mut self.config.visualizer_opacity, "Visualizer Opacity", DEFAULT_VISUALIZER_OPACITY);
        help_text(ui, self.config.show_help_text, "0 = invisible, 255 = fully opaque. Applies to bars, peaks, and outlines.");

        checkbox_reset(ui, &mut self.config.visualizer_outline_enabled, "Enable Outlines", false);
        if self.config.visualizer_outline_enabled {
            slider_f32(ui, &mut self.config.visualizer_outline_thickness, 0.5..=10.0, "Outline Thickness", default_visualizer_outline_thickness());
            color_edit_reset(ui, &mut self.config.visualizer_outline_color, "Outline Color", default_visualizer_outline_color());
            help_text(ui, self.config.show_help_text, "Outlines follow the visualizer's shape. In Smooth mode, a single perimeter is drawn.");
        }
    }

    // ── Colors tab ───────────────────────────────────────────

    fn tab_colors(&mut self, ui: &mut egui::Ui) {
        use cfg::*;

        ui.horizontal_wrapped(|ui| {
            let subtabs = [
                (ColorSubTab::Single,   "Single",   ColorMode::StaticSingle),
                (ColorSubTab::Multi,    "Multi",    ColorMode::StaticMultiple),
                (ColorSubTab::Gradient, "Gradient", ColorMode::Gradient),
                (ColorSubTab::Animated, "Animated", ColorMode::Animated),
            ];
            for (sub, label, mode) in subtabs {
                let selected = self.color_subtab == sub;
                let mut text = egui::RichText::new(label).size(13.0);
                if selected { text = text.strong().color(accent()); }
                if ui.add(egui::SelectableLabel::new(selected, text)).clicked() {
                    self.color_subtab = sub;
                    self.config.color_mode = mode;
                }
                if selected { self.config.color_mode = mode; }
            }
        });
        ui.add_space(4.0);
        ui.separator();

        match self.color_subtab {
            ColorSubTab::Single   => self.color_sub_single(ui),
            ColorSubTab::Multi    => self.color_sub_multi(ui),
            ColorSubTab::Gradient => self.color_sub_gradient(ui),
            ColorSubTab::Animated => self.color_sub_animated(ui),
        }
    }

    fn color_sub_single(&mut self, ui: &mut egui::Ui) {
        let is_dynamic = self.config.static_single_source.is_dynamic();
        ui.horizontal(|ui| {
            if ui.selectable_label(!is_dynamic, "Static").clicked() && is_dynamic {
                let colors = self.active_palettes.static_single_colors.clone();
                self.config.static_single_source = ColorSource::Static(if colors.is_empty() { vec![[255, 255, 255]] } else { colors });
            }
            if ui.selectable_label(is_dynamic, "Dynamic (Album Art)").clicked() && !is_dynamic {
                self.config.static_single_source = ColorSource::Dynamic { palette_size: 1, variants_per_color: 0, priority: PalettePriority::MostCommon };
            }
        });
        if is_dynamic {
            if let ColorSource::Dynamic { priority, .. } = &mut self.config.static_single_source {
                Self::draw_dynamic_palette_settings(ui, "ss", &mut 1u8, &mut 0u8, priority, false, false);
            }
            Self::draw_palette_preview(ui, &self.active_palettes.static_single_colors);
            help_text(ui, self.config.show_help_text, "Single color picked from the album art using the selected priority.");
        } else {
            if let Some(colors) = self.config.static_single_source.static_colors_mut() {
                if colors.is_empty() { colors.push([255, 255, 255]); }
                color_edit_reset(ui, &mut colors[0], "Bar Color", [255, 255, 255]);
            }
            ui.label("Presets:");
            let mut chosen: Option<[u8; 3]> = None;
            let mut to_remove: Option<usize> = None;
            for (chunk_idx, chunk) in self.config.color_presets.chunks(12).enumerate() {
                ui.horizontal(|ui| {
                    for (i, preset) in chunk.iter().enumerate() {
                        let global_i = chunk_idx * 12 + i;
                        let (rect, response) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
                        if ui.is_rect_visible(rect) {
                            let color = egui::Color32::from_rgb(preset.color[0], preset.color[1], preset.color[2]);
                            ui.painter().rect_filled(rect, 4.0, color);
                            if response.clicked() { chosen = Some(preset.color); }
                        }
                        if !preset.is_premade {
                            let btn_rect = egui::Rect::from_center_size(rect.right_top() + egui::vec2(-2.0, 2.0), egui::vec2(16.0, 16.0));
                            let btn = ui.put(btn_rect, egui::Button::new("✖").frame(false).small());
                            if btn.clicked() { to_remove = Some(global_i); }
                        }
                    }
                });
            }
            if let Some(i) = to_remove { self.config.color_presets.remove(i); }
            if let Some(c) = chosen { self.config.static_single_source = ColorSource::Static(vec![c]); }

            if ui.button("💾 Save Current Color as Preset").clicked() {
                if let Some(colors) = self.config.static_single_source.static_colors() {
                    if let Some(c) = colors.first().copied() {
                        if !self.config.color_presets.iter().any(|p| p.color == c) {
                            self.config.color_presets.push(ColorPreset { color: c, is_premade: false });
                        }
                    }
                }
            }
        }
    }

    fn color_sub_multi(&mut self, ui: &mut egui::Ui) {
        let is_dynamic = self.config.static_multi_source.is_dynamic();
        ui.horizontal(|ui| {
            if ui.selectable_label(!is_dynamic, "Static").clicked() && is_dynamic {
                let colors = self.active_palettes.static_multi_colors.clone();
                self.config.static_multi_source = ColorSource::Static(if colors.is_empty() { vec![[255, 255, 255]] } else { colors });
            }
            if ui.selectable_label(is_dynamic, "Dynamic (Album Art)").clicked() && !is_dynamic {
                self.config.static_multi_source = ColorSource::Dynamic { palette_size: 3, variants_per_color: 0, priority: PalettePriority::MostCommon };
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.config.static_mode, StaticDistributionMode::Blocks, "Blocks");
            ui.radio_value(&mut self.config.static_mode, StaticDistributionMode::Sequential, "Sequential");
            ui.radio_value(&mut self.config.static_mode, StaticDistributionMode::Random, "Random");
            ui.radio_value(&mut self.config.static_mode, StaticDistributionMode::Vertical, "Vertical");
        });
        if self.config.static_mode == StaticDistributionMode::Sequential {
            ui.horizontal(|ui| {
                let mut n = self.config.sequential_bars_per_color as i32;
                let r = ui.add(egui::Slider::new(&mut n, 1..=20).text("Bars per Color"));
                if r.secondary_clicked() { n = 1; }
                self.config.sequential_bars_per_color = n.max(1) as usize;
                reset_btn(ui, self.config.sequential_bars_per_color != 1,
                          || self.config.sequential_bars_per_color = 1);
            });
        }
        if self.config.static_mode == StaticDistributionMode::Vertical {
            ui.horizontal(|ui| {
                ui.label("Vertical Color Style:");
                ui.radio_value(&mut self.config.vertical_color_style, VerticalColorStyle::Gradient, "Gradient");
                ui.radio_value(&mut self.config.vertical_color_style, VerticalColorStyle::Blocks, "Blocks");
            });
            help_text(ui, self.config.show_help_text, self.config.vertical_color_style.description());
        }
        ui.add_space(4.0);
        if is_dynamic {
            if let ColorSource::Dynamic { palette_size, variants_per_color, priority } = &mut self.config.static_multi_source {
                Self::draw_dynamic_palette_settings(ui, "sm", palette_size, variants_per_color, priority, true, true);
            }
            Self::draw_palette_preview(ui, &self.active_palettes.static_multi_colors);
        } else {
            let mut to_remove = None;
            let num_colors = self.config.static_multi_source.static_colors().map(|c| c.len()).unwrap_or(0);
            if let Some(colors) = self.config.static_multi_source.static_colors_mut() {
                for (i, color) in colors.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.color_edit_button_srgb(color);
                        if num_colors > 1 && ui.button("❌").clicked() { to_remove = Some(i); }
                    });
                }
            }
            if let Some(i) = to_remove { if let Some(colors) = self.config.static_multi_source.static_colors_mut() { colors.remove(i); } }
            if ui.button("➕ Add Color").clicked() { if let Some(colors) = self.config.static_multi_source.static_colors_mut() { colors.push([255, 255, 255]); } }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.new_preset_name);
                if ui.button("💾 Save Preset").clicked() {
                    let name = if self.new_preset_name.trim().is_empty() { format!("Preset {}", self.config.static_multi_presets.len() + 1) } else { self.new_preset_name.clone() };
                    let colors = self.config.static_multi_source.static_colors().map(|c| c.to_vec()).unwrap_or_default();
                    self.config.static_multi_presets.push(StaticMultiPreset {
                        name, colors, mode: self.config.static_mode,
                        bars_per_color: self.config.sequential_bars_per_color, is_premade: false,
                    });
                    self.new_preset_name.clear();
                }
            });
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(30, 30, 30))
                .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                .rounding(4.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Left-Click").strong().color(accent()));
                        ui.label("= Colors only  |  ");
                        ui.label(egui::RichText::new("Right-Click").strong().color(accent()));
                        ui.label("= All settings");
                    });
                });
            ui.add_space(4.0);
            
            let mut apply_colors: Option<usize> = None;
            let mut apply_full: Option<usize> = None;
            let mut remove_idx: Option<usize> = None;
            ui.horizontal_wrapped(|ui| {
                for (i, preset) in self.config.static_multi_presets.iter().enumerate() {
                    let mut btn = ui.button(&preset.name);
                    if btn.clicked() { apply_colors = Some(i); }
                    if btn.secondary_clicked() { apply_full = Some(i); }
                    btn = btn.on_hover_text("Left: colors only · Right: colors + mode");
                    let _ = btn;
                    if !preset.is_premade && ui.small_button("✖").clicked() { remove_idx = Some(i); }
                }
            });
            if let Some(i) = apply_colors {
                let p = &self.config.static_multi_presets[i];
                self.config.static_multi_source = ColorSource::Static(p.colors.clone());
            }
            if let Some(i) = apply_full {
                let p = &self.config.static_multi_presets[i];
                self.config.static_multi_source = ColorSource::Static(p.colors.clone());
                self.config.static_mode = p.mode;
                self.config.sequential_bars_per_color = p.bars_per_color;
            }
            if let Some(i) = remove_idx { if !self.config.static_multi_presets[i].is_premade { self.config.static_multi_presets.remove(i); } }
        }
    }

    fn color_sub_gradient(&mut self, ui: &mut egui::Ui) {
        let is_dynamic = self.config.gradient_source.is_dynamic();
        ui.horizontal(|ui| {
            if ui.selectable_label(!is_dynamic, "Static").clicked() && is_dynamic {
                let stops = self.active_palettes.gradient_stops.clone();
                self.config.gradient_source = GradientSource::Static(
                    if stops.len() >= 2 { stops }
                    else if stops.len() == 1 { vec![stops[0].clone(), GradientStop { color: stops[0].color, position: 1.0 }] }
                    else { vec![GradientStop { color: [255, 0, 0], position: 0.0 }, GradientStop { color: [0, 0, 255], position: 1.0 }] }
                );
            }
            if ui.selectable_label(is_dynamic, "Dynamic (Album Art)").clicked() && !is_dynamic {
                self.config.gradient_source = GradientSource::Dynamic { palette_size: 2, variants_per_color: 0, priority: PalettePriority::MostCommon };
            }
        });
        slider_f32(ui, &mut self.config.gradient_angle, 0.0..=360.0, "Angle (°)", 0.0);
        help_text(ui, self.config.show_help_text, "0° = horizontal, 90° = vertical, 45° = diagonal.");
        if is_dynamic {
            if let GradientSource::Dynamic { palette_size, variants_per_color, priority } = &mut self.config.gradient_source {
                Self::draw_dynamic_palette_settings(ui, "g", palette_size, variants_per_color, priority, true, true);
            }
            let preview: Vec<[u8; 3]> = self.active_palettes.gradient_stops.iter().map(|s| s.color).collect();
            Self::draw_palette_preview(ui, &preview);
        } else {
            let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width() - 16.0, 30.0), egui::Sense::click_and_drag());
            let painter = ui.painter();
            let preview_stops = self.config.gradient_source.static_stops().map(|s| s.to_vec()).unwrap_or_default();
            let steps = 100;
            for i in 0..steps {
                let t1 = i as f32 / steps as f32;
                let t2 = (i + 1) as f32 / steps as f32;
                let c1 = color::sample_gradient(&preview_stops, t1);
                let x1 = rect.left() + rect.width() * t1;
                let x2 = rect.left() + rect.width() * t2;
                let r = egui::Rect::from_min_max(egui::pos2(x1, rect.top()), egui::pos2(x2, rect.bottom()));
                painter.rect_filled(r, 0.0, egui::Color32::from_rgb(c1[0], c1[1], c1[2]));
            }
            let mut sort_stops = false;
            if let Some(stops) = self.config.gradient_source.static_stops_mut() {
                for (i, stop) in stops.iter_mut().enumerate() {
                    let x = rect.left() + rect.width() * stop.position;
                    let stop_rect = egui::Rect::from_center_size(egui::pos2(x, rect.center().y), egui::vec2(12.0, 30.0));
                    let stop_response = ui.interact(stop_rect, ui.id().with("gradient_stop").with(i), egui::Sense::click_and_drag());
                    if stop_response.dragged() {
                        stop.position = (stop.position + stop_response.drag_delta().x / rect.width()).clamp(0.0, 1.0);
                        sort_stops = true;
                    }
                    let handle_color = egui::Color32::from_rgb(stop.color[0], stop.color[1], stop.color[2]);
                    painter.rect_filled(stop_rect, 2.0, handle_color);
                    painter.rect_stroke(stop_rect, 2.0_f32, egui::Stroke::new(2.0_f32, egui::Color32::WHITE));
                }
            }
            if sort_stops { if let Some(stops) = self.config.gradient_source.static_stops_mut() { stops.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap()); } }
            if response.double_clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                    let c = color::sample_gradient(&preview_stops, t);
                    if let Some(stops) = self.config.gradient_source.static_stops_mut() {
                        stops.push(GradientStop { color: c, position: t });
                        stops.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
                    }
                }
            }
            ui.label("Double-click the bar to add a color stop.");
            let mut to_remove = None;
            let current_len = self.config.gradient_source.static_stops().map(|s| s.len()).unwrap_or(0);
            if let Some(stops) = self.config.gradient_source.static_stops_mut() {
                for (i, stop) in stops.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.color_edit_button_srgb(&mut stop.color);
                        ui.label(format!("Pos: {:.2}", stop.position));
                        if current_len > 2 && ui.button("❌").clicked() { to_remove = Some(i); }
                    });
                }
            }
            if let Some(i) = to_remove { if let Some(stops) = self.config.gradient_source.static_stops_mut() { stops.remove(i); } }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.new_preset_name);
                if ui.button("💾 Save Preset").clicked() {
                    let name = if self.new_preset_name.trim().is_empty() { format!("Preset {}", self.config.gradient_presets.len() + 1) } else { self.new_preset_name.clone() };
                    let stops = self.config.gradient_source.static_stops().map(|s| s.to_vec()).unwrap_or_default();
                    self.config.gradient_presets.push(GradientPreset {
                        name, stops, angle: self.config.gradient_angle, is_premade: false,
                    });
                    self.new_preset_name.clear();
                }
            });
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(30, 30, 30))
                .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                .rounding(4.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Left-Click").strong().color(accent()));
                        ui.label("= Colors only  |  ");
                        ui.label(egui::RichText::new("Right-Click").strong().color(accent()));
                        ui.label("= All settings");
                    });
                });
            ui.add_space(4.0);

            let mut apply_colors: Option<usize> = None;
            let mut apply_full: Option<usize> = None;
            let mut remove_idx: Option<usize> = None;
            ui.horizontal_wrapped(|ui| {
                for (i, preset) in self.config.gradient_presets.iter().enumerate() {
                    let btn = ui.button(&preset.name);
                    if btn.clicked() { apply_colors = Some(i); }
                    if btn.secondary_clicked() { apply_full = Some(i); }
                    btn.on_hover_text("Left: colors only · Right: colors + angle");
                    if !preset.is_premade && ui.small_button("✖").clicked() { remove_idx = Some(i); }
                }
            });
            if let Some(i) = apply_colors {
                let p = &self.config.gradient_presets[i];
                self.config.gradient_source = GradientSource::Static(p.stops.clone());
            }
            if let Some(i) = apply_full {
                let p = &self.config.gradient_presets[i];
                self.config.gradient_source = GradientSource::Static(p.stops.clone());
                self.config.gradient_angle = p.angle;
            }
            if let Some(i) = remove_idx { if !self.config.gradient_presets[i].is_premade { self.config.gradient_presets.remove(i); } }
        }
    }

    fn color_sub_animated(&mut self, ui: &mut egui::Ui) {
        let is_dynamic = self.config.animated_source.is_dynamic();
        ui.horizontal(|ui| {
            if ui.selectable_label(!is_dynamic, "Static").clicked() && is_dynamic {
                let colors = self.active_palettes.animated_colors.clone();
                self.config.animated_source = ColorSource::Static(if colors.is_empty() { vec![[255, 255, 255]] } else { colors });
            }
            if ui.selectable_label(is_dynamic, "Dynamic (Album Art)").clicked() && !is_dynamic {
                self.config.animated_source = ColorSource::Dynamic { palette_size: 3, variants_per_color: 0, priority: PalettePriority::MostCommon };
            }
        });
        if is_dynamic {
            if let ColorSource::Dynamic { palette_size, variants_per_color, priority } = &mut self.config.animated_source {
                Self::draw_dynamic_palette_settings(ui, "anim", palette_size, variants_per_color, priority, true, true);
            }
            Self::draw_palette_preview(ui, &self.active_palettes.animated_colors);
        } else {
            let mut to_remove = None;
            let current_len = self.config.animated_source.static_colors().map(|c| c.len()).unwrap_or(0);
            if let Some(colors) = self.config.animated_source.static_colors_mut() {
                for (i, color) in colors.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.color_edit_button_srgb(color);
                        if current_len > 1 && ui.button("❌").clicked() { to_remove = Some(i); }
                    });
                }
            }
            if let Some(i) = to_remove { if let Some(colors) = self.config.animated_source.static_colors_mut() { colors.remove(i); } }
            if ui.button("➕ Add Color").clicked() { if let Some(colors) = self.config.animated_source.static_colors_mut() { colors.push([255, 255, 255]); } }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.new_preset_name);
                if ui.button("💾 Save Preset").clicked() {
                    let name = if self.new_preset_name.trim().is_empty() { format!("Preset {}", self.config.color_change_presets.len() + 1) } else { self.new_preset_name.clone() };
                    let colors = self.config.animated_source.static_colors().map(|c| c.to_vec()).unwrap_or_default();
                    self.config.color_change_presets.push(ColorChangePreset {
                        name, colors, speed: self.config.color_change_speed,
                        is_premade: false, motion: self.config.color_change_motion.clone(),
                    });
                    self.new_preset_name.clear();
                }
            });
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(30, 30, 30))
                .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                .rounding(4.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Left-Click").strong().color(accent()));
                        ui.label("= Colors only  |  ");
                        ui.label(egui::RichText::new("Right-Click").strong().color(accent()));
                        ui.label("= All settings");
                    });
                });
            ui.add_space(4.0);

            let mut apply_colors: Option<usize> = None;
            let mut apply_full: Option<usize> = None;
            let mut remove_idx: Option<usize> = None;
            ui.horizontal_wrapped(|ui| {
                for (i, preset) in self.config.color_change_presets.iter().enumerate() {
                    let btn = ui.button(&preset.name);
                    if btn.clicked() { apply_colors = Some(i); }
                    if btn.secondary_clicked() { apply_full = Some(i); }
                    btn.on_hover_text("Left: colors only · Right: colors + speed + motion");
                    if !preset.is_premade && ui.small_button("✖").clicked() { remove_idx = Some(i); }
                }
            });
            if let Some(i) = apply_colors {
                self.config.animated_source = ColorSource::Static(self.config.color_change_presets[i].colors.clone());
            }
            if let Some(i) = apply_full {
                let p = self.config.color_change_presets[i].clone();
                self.config.animated_source = ColorSource::Static(p.colors);
                self.config.color_change_speed = p.speed;
                self.config.color_change_motion = p.motion;
            }
            if let Some(i) = remove_idx { if !self.config.color_change_presets[i].is_premade { self.config.color_change_presets.remove(i); } }
        }
        slider_f32_scale(ui, &mut self.config.color_change_speed, 0.0..=1.0, "Speed", 0.5);
        
        section_header(ui, "Motion Shape");
        combo_reset(ui, &mut self.config.color_change_motion.shape, MotionShape::default(),
                    "Shape", "motion_shape", 180.0,
                    &MotionShape::all(), MotionShape::display_name);
        if self.config.color_change_motion.shape.uses_vertical_axis() {
            ui.horizontal(|ui| {
                ui.label("Color Style:");
                ui.radio_value(&mut self.config.vertical_color_style, VerticalColorStyle::Gradient, "Gradient");
                ui.radio_value(&mut self.config.vertical_color_style, VerticalColorStyle::Blocks, "Blocks");
            });
            help_text(ui, self.config.show_help_text, self.config.vertical_color_style.description());
        }

        section_header(ui, "Blend");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.config.color_change_motion.blend, BlendMode::Smooth, "Smooth");
            ui.radio_value(&mut self.config.color_change_motion.blend, BlendMode::Hard, "Hard");
        });
        help_text(ui, self.config.show_help_text, "Smooth = gradients / soft fades. Hard = sharp on/off edges.");

        section_header(ui, "Motion Speed");
        slider_f32(ui, &mut self.config.color_change_motion.speed, 0.0..=2.0, "Motion Speed", 1.0);

        let shape = self.config.color_change_motion.shape;
        let hard = self.config.color_change_motion.blend == BlendMode::Hard;
        if matches!(shape, MotionShape::SineWave | MotionShape::TriangleWave | MotionShape::SquareWave | MotionShape::SawtoothWave | MotionShape::WaveInterference) {
            slider_f32(ui, &mut self.config.color_change_motion.wave_frequency, 0.5..=8.0, "Wave Frequency", 2.0);
        }
        if matches!(shape, MotionShape::Comet | MotionShape::BouncingBall | MotionShape::Sinelon) {
            slider_f32(ui, &mut self.config.color_change_motion.trail_length, 0.05..=0.9, "Trail Length", 0.25);
        }
        if matches!(shape, MotionShape::Stutter | MotionShape::Sparkle | MotionShape::Confetti) {
            slider_f32(ui, &mut self.config.color_change_motion.stutter_intensity, 0.0..=1.0, "Event Intensity", 0.3);
        }
        if matches!(shape, MotionShape::HorizontalWipe | MotionShape::VerticalWipe | MotionShape::DiagonalWipe | MotionShape::RadialWipe) && !hard {
            slider_f32(ui, &mut self.config.color_change_motion.wipe_softness, 0.01..=0.5, "Wipe Softness", 0.1);
        }
        if shape.uses_intensity() {
            color_edit_reset(ui, &mut self.config.color_change_motion.off_color, "Off Color", [0, 0, 0]);
            slider_u8(ui, &mut self.config.color_change_motion.off_color_opacity, "Off-Color Opacity", 255);
            help_text(ui, self.config.show_help_text, "Colour that 'off' pixels fade toward.");
        }
    }

    // ── Audio tab ────────────────────────────────────────────

    fn tab_audio(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
        let show_help = self.config.show_help_text;

        section_header(ui, "Presets");
        self.draw_audio_preset_selector(ui);

        let a = &mut self.config.audio;

        section_header(ui, "Signal Processing");
        fft_size_combo(ui, a);
        
        combo_reset(ui, &mut a.window_function, WindowFunction::default(), "Window Function", "window_fn", 160.0, &WindowFunction::all(), WindowFunction::display_name);
        help_text(ui, show_help, a.window_function.description());

        combo_reset(ui, &mut a.frequency_scaling, FrequencyScaling::default(), "Frequency Scaling", "freq_scaling", 160.0, &[FrequencyScaling::Linear, FrequencyScaling::Logarithmic], FrequencyScaling::display_name);
        help_text(ui, show_help, a.frequency_scaling.description());

        combo_reset(ui, &mut a.channel_mix, ChannelMixMode::default(), "Channel Mix", "channel_mix", 160.0, &ChannelMixMode::all(), ChannelMixMode::display_name);
        help_text(ui, show_help, a.channel_mix.description());

        combo_reset(ui, &mut a.weighting, WeightingCurve::default(), "Weighting Curve", "weighting", 180.0, &WeightingCurve::all(), WeightingCurve::display_name);
        help_text(ui, show_help, a.weighting.description());

        checkbox_reset(ui, &mut a.dc_block, "DC Block / High-Pass Filter", DEFAULT_DC_BLOCK);
        help_text(ui, show_help, "Filters out DC offset and sub-10 Hz rumble before analysis.");

        slider_f32(ui, &mut a.frequency_smoothing, 0.0..=1.0, "Frequency Smoothing", DEFAULT_FREQ_SMOOTHING);
        help_text(ui, show_help, "Averages adjacent bars. Higher = fewer spiky, jittery bars.");

        checkbox_reset(ui, &mut a.harmonic_emphasis, "Harmonic Emphasis", DEFAULT_HARMONIC_EMPHASIS);
        help_text(ui, show_help, "Blooms local spectral peaks into wider blobs. Helps with solo piano and sparse material.");

        ui.horizontal(|ui| {
            ui.label("Min Frequency:");
            let r = ui.add(egui::DragValue::new(&mut a.min_frequency_hz).speed(10.0).clamp_range(10.0..=1000.0).suffix(" Hz"));
            if r.secondary_clicked() { a.min_frequency_hz = DEFAULT_MIN_FREQ; }
            reset_btn(ui, (a.min_frequency_hz - DEFAULT_MIN_FREQ).abs() > 1e-3, || a.min_frequency_hz = DEFAULT_MIN_FREQ);
        });
        ui.horizontal(|ui| {
            ui.label("Max Frequency:");
            let r = ui.add(egui::DragValue::new(&mut a.max_frequency_hz).speed(100.0).clamp_range(2000.0..=24000.0).suffix(" Hz"));
            if r.secondary_clicked() { a.max_frequency_hz = DEFAULT_MAX_FREQ; }
            reset_btn(ui, (a.max_frequency_hz - DEFAULT_MAX_FREQ).abs() > 1e-3, || a.max_frequency_hz = DEFAULT_MAX_FREQ);
        });

        section_header(ui, "Amplitude");
        slider_f32(ui, &mut a.sensitivity, 0.0..=100.0, "Sensitivity", DEFAULT_SENSITIVITY);
        help_text(ui, show_help, "Primary reactivity slider. Drives raw input gain and lowers the noise floor.");

        slider_f32(ui, &mut a.raw_gain, 0.25..=4.0, "Raw Gain", DEFAULT_RAW_GAIN);
        help_text(ui, show_help, "Direct pre-analysis multiplier. Normally driven by Sensitivity.");

        slider_f32(ui, &mut a.response_gamma, 0.5..=2.5, "Response Curve (γ)", DEFAULT_RESPONSE_GAMMA);
        help_text(ui, show_help, "<1 lifts quiet passages; >1 emphasises peaks. Default 1.5 = balanced.");

        slider_f32(ui, &mut a.noise_gate, 0.0..=0.5, "Noise Gate", DEFAULT_NOISE_GATE);
        help_text(ui, show_help, "Bars below this threshold are snapped to 0. Removes hovering bars during silence.");

        ui.horizontal(|ui| {
            ui.label("Min dB:");
            let r = ui.add(egui::Slider::new(&mut a.min_decibels, -120.0..=-30.0).show_value(true));
            if r.secondary_clicked() { a.min_decibels = DEFAULT_MIN_DB; }
            reset_btn(ui, (a.min_decibels - DEFAULT_MIN_DB).abs() > 1e-3, || a.min_decibels = DEFAULT_MIN_DB);
        });
        ui.horizontal(|ui| {
            ui.label("Max dB:");
            let r = ui.add(egui::Slider::new(&mut a.max_decibels, -60.0..=0.0).show_value(true));
            if r.secondary_clicked() { a.max_decibels = DEFAULT_MAX_DB; }
            reset_btn(ui, (a.max_decibels - DEFAULT_MAX_DB).abs() > 1e-3, || a.max_decibels = DEFAULT_MAX_DB);
        });
        if a.min_decibels >= a.max_decibels {
            ui.colored_label(egui::Color32::RED, "⚠ Max dB must be greater than Min dB.");
        }
        slider_f32(ui, &mut a.target_dynamic_range_db, 20.0..=90.0, "Target Dynamic Range (dB)", DEFAULT_TARGET_RANGE);
        help_text(ui, show_help, "Contrast between quiet and loud passages. Smaller = more compression/punch.");

        checkbox_reset(ui, &mut a.per_frame_normalization, "Per-Frame Peak Normalization", DEFAULT_PER_FRAME_NORMALIZATION);
        help_text(ui, show_help, "Scales each frame so the loudest bar hits the top. Makes quiet tracks visible.");

        checkbox_reset(ui, &mut a.agc_enabled, "AGC / Peak Normalisation", DEFAULT_AGC_ENABLED);
        help_text(ui, show_help, "Slowly normalises track loudness so quiet songs stay lively.");
        if a.agc_enabled {
            slider_f32(ui, &mut a.agc_attack, 0.01..=0.9, "AGC Attack", DEFAULT_AGC_ATTACK);
            slider_f32(ui, &mut a.agc_release, 0.0001..=0.05, "AGC Release", DEFAULT_AGC_RELEASE);
        }

        section_header(ui, "Frequency Shaping");
        slider_f32(ui, &mut a.eq_bass_db, -10.0..=10.0, "Bass EQ (< 250 Hz)", DEFAULT_EQ_BASS);
        slider_f32(ui, &mut a.eq_mid_db, -10.0..=10.0, "Mid EQ (250–4k Hz)", DEFAULT_EQ_MID);
        slider_f32(ui, &mut a.eq_treble_db, -10.0..=10.0, "Treble EQ (> 4k Hz)", DEFAULT_EQ_TREBLE);
        slider_f32(ui, &mut a.spectrum_tilt, -6.0..=6.0, "Spectrum Tilt (dB/oct)", DEFAULT_SPECTRUM_TILT);
        help_text(ui, show_help, "Balances bass and treble. Positive values boost highs and tame bass.");

        section_header(ui, "Timing");
        slider_f32(ui, &mut a.poll_rate_hz, 10.0..=120.0, "Poll Rate (Hz)", DEFAULT_POLL_RATE);
        
        ui.horizontal(|ui| {
            let mut n = a.average_size as i32;
            let r = ui.add(egui::Slider::new(&mut n, 1..=16).text("Average Size (frames)"));
            if r.secondary_clicked() { n = DEFAULT_AVERAGE_SIZE as i32; }
            a.average_size = n.max(1) as usize;
            reset_btn(ui, a.average_size != DEFAULT_AVERAGE_SIZE, || a.average_size = DEFAULT_AVERAGE_SIZE);
        });

        section_header(ui, "Ballistics");
        if a.decay_mode != self.last_decay_mode {
            let (atk, dec) = a.decay_mode.default_ballistics();
            a.attack_time_ms = atk;
            a.decay_time_ms = dec;
            self.last_decay_mode = a.decay_mode;
        }
        combo_reset(ui, &mut a.decay_mode, DecayMode::default(), "Decay Mode", "decay_mode", 160.0, &[DecayMode::Linear, DecayMode::Exponential, DecayMode::Gravity], DecayMode::display_name);
        help_text(ui, show_help, a.decay_mode.description());
        
        slider_f32(ui, &mut a.attack_time_ms, 0.0..=100.0, "Attack Time (ms)", DEFAULT_ATTACK_MS);
        slider_f32(ui, &mut a.decay_time_ms, 50.0..=2000.0, "Decay Time (ms)", DEFAULT_DECAY_MS);
        ui.horizontal(|ui| {
            ui.label("Peak Hold (ms):");
            let r = ui.add(egui::Slider::new(&mut a.peak_hold_ms, 0.0..=2000.0));
            if r.secondary_clicked() { a.peak_hold_ms = DEFAULT_PEAK_HOLD_MS; }
            reset_btn(ui, (a.peak_hold_ms - DEFAULT_PEAK_HOLD_MS).abs() > 1e-3, || a.peak_hold_ms = DEFAULT_PEAK_HOLD_MS);
        });
        if a.decay_mode == DecayMode::Gravity || a.peak_hold_ms > 0.0 {
            slider_f32(ui, &mut a.gravity_coefficient, 0.1..=5.0, "Gravity Coefficient", DEFAULT_GRAVITY_COEFF);
        }
        checkbox_reset(ui, &mut a.per_band_ballistics, "Per-Band Ballistics", DEFAULT_PER_BAND_BALLISTICS);
        help_text(ui, show_help, "Lets bass / mid / treble use different attack and decay times.");
        if a.per_band_ballistics {
            ui.label("Bass:");
            slider_f32(ui, &mut a.bass_attack_ms, 0.0..=100.0, "  Attack (ms)", DEFAULT_ATTACK_MS);
            slider_f32(ui, &mut a.bass_decay_ms, 50.0..=2000.0, "  Decay (ms)", DEFAULT_DECAY_MS);
            ui.label("Mid:");
            slider_f32(ui, &mut a.mid_attack_ms, 0.0..=100.0, "  Attack (ms)", DEFAULT_ATTACK_MS);
            slider_f32(ui, &mut a.mid_decay_ms, 50.0..=2000.0, "  Decay (ms)", DEFAULT_DECAY_MS);
            ui.label("Treble:");
            slider_f32(ui, &mut a.treble_attack_ms, 0.0..=100.0, "  Attack (ms)", DEFAULT_ATTACK_MS);
            slider_f32(ui, &mut a.treble_decay_ms, 50.0..=2000.0, "  Decay (ms)", DEFAULT_DECAY_MS);
        }

        section_header(ui, "Rhythm");
        combo_reset(ui, &mut a.beat_focus, BeatFocusMode::default(), "Beat Focus", "beat_focus", 160.0, &BeatFocusMode::all(), BeatFocusMode::display_name);
        help_text(ui, show_help, a.beat_focus.description());
        
        slider_f32(ui, &mut a.beat_detection_sensitivity, 0.0..=1.0, "Beat Detection Sensitivity", DEFAULT_BEAT_SENS);
        slider_f32(ui, &mut a.energy_threshold_multiplier, 1.0..=3.0, "Energy Threshold Multiplier", DEFAULT_ENERGY_THRESHOLD);
        slider_f32(ui, &mut a.beat_envelope_decay_ms, 20.0..=500.0, "Beat Envelope Decay (ms)", DEFAULT_BEAT_ENVELOPE_DECAY_MS);
        help_text(ui, show_help, "How quickly the beat flash fades out after a detection.");
        
        checkbox_reset(ui, &mut a.bpm_lock_enabled, "BPM Lock / Phase Smoothing", false);
        if a.bpm_lock_enabled {
            slider_f32(ui, &mut a.bpm_phase_smoothing, 0.0..=1.0, "Phase Smoothing", DEFAULT_BPM_PHASE_SMOOTHING);
        }
        
        let _ = &a;
        if self.last_bpm > 0.0 {
            ui.label(format!("Detected BPM: {:.1}", self.last_bpm));
        }
    }

    fn draw_audio_preset_selector(&mut self, ui: &mut egui::Ui) {
        let detected = AudioPreset::detect(&self.config.audio);
        let custom_match = self.config.custom_audio_presets.iter()
            .find(|p| p.settings == self.config.audio);

        let current_label = if let Some(p) = detected {
            p.display_name().to_string()
        } else if let Some(p) = custom_match {
            p.name.clone()
        } else {
            "Custom".to_string()
        };

        let mut chosen_builtin: Option<AudioPreset> = None;
        let mut chosen_custom: Option<usize> = None;

        ui.horizontal(|ui| {
            ui.label("Preset:");
            egui::ComboBox::from_id_source("audio_preset")
                .selected_text(current_label)
                .width(200.0)
                .show_ui(ui, |ui| {
                    for p in AudioPreset::all() {
                        let active = detected == Some(p);
                        if ui.selectable_label(active, p.display_name())
                            .on_hover_text(p.description())
                            .clicked()
                        {
                            chosen_builtin = Some(p);
                        }
                    }
                    if !self.config.custom_audio_presets.is_empty() {
                        ui.separator();
                        for (i, p) in self.config.custom_audio_presets.iter().enumerate() {
                            let active = custom_match.map(|c| c.name == p.name).unwrap_or(false);
                            if ui.selectable_label(active, &p.name).clicked() {
                                chosen_custom = Some(i);
                            }
                        }
                    }
                });
        });
        if let Some(p) = chosen_builtin {
            p.apply(&mut self.config.audio);
            self.last_decay_mode = self.config.audio.decay_mode;
            self.audio_dirty = true;
        }
        if let Some(i) = chosen_custom {
            self.config.audio = self.config.custom_audio_presets[i].settings.clone();
            self.last_decay_mode = self.config.audio.decay_mode;
            self.audio_dirty = true;
        }
        if let Some(p) = detected {
            help_text(ui, self.config.show_help_text, p.description());
        }

        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.new_audio_preset_name);
            if ui.button("💾 Save as Preset").clicked() {
                let name = if self.new_audio_preset_name.trim().is_empty() {
                    format!("My Preset {}", self.config.custom_audio_presets.len() + 1)
                } else {
                    self.new_audio_preset_name.trim().to_string()
                };
                self.config.custom_audio_presets.push(CustomAudioPreset {
                    name,
                    settings: self.config.audio.clone(),
                });
                self.new_audio_preset_name.clear();
            }
        });

        let mut to_remove = None;
        let custom_list: Vec<(usize, String)> = self.config.custom_audio_presets.iter()
            .enumerate().map(|(i, p)| (i, p.name.clone())).collect();
        ui.horizontal_wrapped(|ui| {
            for (i, name) in &custom_list {
                if ui.button(name).clicked() {
                    self.config.audio = self.config.custom_audio_presets[*i].settings.clone();
                    self.last_decay_mode = self.config.audio.decay_mode;
                    self.audio_dirty = true;
                }
                if ui.small_button("✖").clicked() { to_remove = Some(*i); }
            }
        });
        if let Some(i) = to_remove { self.pending_audio_preset_delete = Some(i); }
    }

    // ── Track Info tab ───────────────────────────────────────

    fn tab_track_info(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
        // Avoid cloning the font list every frame — borrow it directly.
        let show_help = self.config.show_help_text;
        let system_fonts = &self.system_fonts;
        let cfg = &mut self.config.track_info;

        section_header(ui, "Cover Art");
        checkbox_reset(ui, &mut cfg.show_cover_art, "Show Cover Art", true);
        if cfg.show_cover_art {
            ui.horizontal(|ui| {
                let r = ui.add(egui::Slider::new(&mut cfg.cover_size, 40.0..=600.0).text("Size (px)"));
                if r.secondary_clicked() { cfg.cover_size = DEFAULT_COVER_SIZE; }
                reset_btn(ui, (cfg.cover_size - DEFAULT_COVER_SIZE).abs() > 1e-3, || cfg.cover_size = DEFAULT_COVER_SIZE);
            });
            ui.horizontal(|ui| {
                ui.label("X Offset:");
                let r = ui.add(egui::DragValue::new(&mut cfg.cover_x_offset).speed(1.0).max_decimals(0).clamp_range(-5000.0..=200.0).suffix(" px"));
                if r.secondary_clicked() { cfg.cover_x_offset = 0.0; }
                reset_btn(ui, (cfg.cover_x_offset).abs() > 1e-3, || cfg.cover_x_offset = 0.0);
            });
            ui.horizontal(|ui| {
                ui.label("Y Offset:");
                let r = ui.add(egui::DragValue::new(&mut cfg.cover_y_offset).speed(1.0).max_decimals(0).clamp_range(-200.0..=5000.0).suffix(" px"));
                if r.secondary_clicked() { cfg.cover_y_offset = 0.0; }
                reset_btn(ui, (cfg.cover_y_offset).abs() > 1e-3, || cfg.cover_y_offset = 0.0);
            });
            slider_u8(ui, &mut cfg.cover_opacity, "Cover Art Opacity", DEFAULT_COVER_OPACITY);
        }

        section_header(ui, "Song Info Text");
        ui.horizontal(|ui| {
            checkbox_reset(ui, &mut cfg.show_title, "Show Title", true);
            checkbox_reset(ui, &mut cfg.show_artist, "Show Artist", true);
        });
        ui.horizontal(|ui| {
            let r = ui.add(egui::Slider::new(&mut cfg.title_font_size, 6.0..=120.0).text("Title Font Size"));
            if r.secondary_clicked() { cfg.title_font_size = DEFAULT_TITLE_FONT_SIZE; }
            reset_btn(ui, (cfg.title_font_size - DEFAULT_TITLE_FONT_SIZE).abs() > 1e-3, || cfg.title_font_size = DEFAULT_TITLE_FONT_SIZE);
        });
        ui.horizontal(|ui| {
            let r = ui.add(egui::Slider::new(&mut cfg.artist_font_size, 6.0..=120.0).text("Artist Font Size"));
            if r.secondary_clicked() { cfg.artist_font_size = DEFAULT_ARTIST_FONT_SIZE; }
            reset_btn(ui, (cfg.artist_font_size - DEFAULT_ARTIST_FONT_SIZE).abs() > 1e-3, || cfg.artist_font_size = DEFAULT_ARTIST_FONT_SIZE);
        });
        ui.horizontal(|ui| {
            ui.label("Font:");
            let selected_label = match &cfg.font_path {
                None => "Default (proportional)".to_string(),
                Some(p) => std::path::Path::new(p).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| p.clone()),
            };
            egui::ComboBox::from_id_source("track_font_select")
                .selected_text(selected_label)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(cfg.font_path.is_none(), "Default (proportional)").clicked() {
                        cfg.font_path = None;
                    }
                    for (name, path) in system_fonts.iter() {
                        let selected = cfg.font_path.as_deref() == Some(path.as_str());
                        if ui.selectable_label(selected, name).clicked() { cfg.font_path = Some(path.clone()); }
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Custom path:");
            let mut path_str = cfg.font_path.clone().unwrap_or_default();
            if ui.text_edit_singleline(&mut path_str).changed() {
                cfg.font_path = if path_str.trim().is_empty() { None } else { Some(path_str) };
            }
        });

        section_header(ui, "Text Color");
        ui.horizontal(|ui| {
            ui.radio_value(&mut cfg.color_mode, TrackTextColorMode::Static, "Static");
            ui.radio_value(&mut cfg.color_mode, TrackTextColorMode::Visualizer, "Visualizer Color");
        });
        if cfg.color_mode == TrackTextColorMode::Static {
            color_edit_reset(ui, &mut cfg.static_color, "Color", [220, 220, 220]);
        }
        slider_u8(ui, &mut cfg.opacity, "Text Opacity", 255);

        section_header(ui, "Text Layout");
        ui.horizontal(|ui| {
            ui.label("Alignment:");
            ui.radio_value(&mut cfg.alignment, TrackTextAlign::Left, "Left");
            ui.radio_value(&mut cfg.alignment, TrackTextAlign::Center, "Center");
            ui.radio_value(&mut cfg.alignment, TrackTextAlign::Right, "Right");
        });
        ui.horizontal(|ui| {
            let r = ui.add(egui::Slider::new(&mut cfg.letter_spacing, -5.0..=20.0).text("Letter Spacing (px)"));
            if r.secondary_clicked() { cfg.letter_spacing = 0.0; }
            reset_btn(ui, cfg.letter_spacing.abs() > 1e-3, || cfg.letter_spacing = 0.0);
        });
        ui.horizontal(|ui| {
            let r = ui.add(egui::Slider::new(&mut cfg.line_height, 0.5..=3.0).text("Line Height"));
            if r.secondary_clicked() { cfg.line_height = 1.3; }
            reset_btn(ui, (cfg.line_height - 1.3).abs() > 1e-3, || cfg.line_height = 1.3);
        });
        ui.horizontal(|ui| {
            ui.label("Text X Offset:");
            let r = ui.add(egui::DragValue::new(&mut cfg.text_x_offset).speed(1.0).max_decimals(0).clamp_range(-5000.0..=200.0).suffix(" px"));
            if r.secondary_clicked() { cfg.text_x_offset = 0.0; }
            reset_btn(ui, cfg.text_x_offset.abs() > 1e-3, || cfg.text_x_offset = 0.0);
        });
        ui.horizontal(|ui| {
            ui.label("Text Y Offset:");
            let r = ui.add(egui::DragValue::new(&mut cfg.text_y_offset).speed(1.0).max_decimals(0).clamp_range(-200.0..=5000.0).suffix(" px"));
            if r.secondary_clicked() { cfg.text_y_offset = 0.0; }
            reset_btn(ui, cfg.text_y_offset.abs() > 1e-3, || cfg.text_y_offset = 0.0);
        });

        section_header(ui, "Shadow");
        checkbox_reset(ui, &mut cfg.shadow_enabled, "Enable Shadow", true);
        if cfg.shadow_enabled {
            slider_f32(ui, &mut cfg.shadow_offset_x, -30.0..=30.0, "Shadow X Offset", 2.0);
            slider_f32(ui, &mut cfg.shadow_offset_y, -30.0..=30.0, "Shadow Y Offset", 2.0);
            slider_f32(ui, &mut cfg.shadow_blur, 0.0..=10.0, "Shadow Blur", 0.0);
            color_edit_reset(ui, &mut cfg.shadow_color, "Shadow Color", [0, 0, 0]);
        }

        section_header(ui, "Outline");
        checkbox_reset(ui, &mut cfg.outline_enabled, "Enable Outline", false);
        if cfg.outline_enabled {
            slider_f32(ui, &mut cfg.outline_thickness, 0.1..=6.0, "Outline Thickness", 1.0);
            color_edit_reset(ui, &mut cfg.outline_color, "Outline Color", [0, 0, 0]);
        }

        let _ = show_help;
    }

    // ── Appearance tab ───────────────────────────────────────

    fn tab_appearance(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
        let show_help = self.config.show_help_text;

        section_header(ui, "Background");
        color_edit_reset(ui, &mut self.config.bg_color, "Background Color", [0, 0, 0]);
        slider_u8(ui, &mut self.config.bg_opacity, "Background Opacity", DEFAULT_BG_OPACITY);
        help_text(ui, show_help, "0 = fully transparent overlay.");

        section_header(ui, "Menu");
        ui.horizontal(|ui| {
            ui.label("Menu Scale:");
            let mut current_scale = self.config.settings_scale;
            egui::ComboBox::from_id_source("menu_scale")
                .selected_text(format!("{:.0}%", current_scale * 100.0))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut current_scale, 0.75, "75%");
                    ui.selectable_value(&mut current_scale, 1.00, "100%");
                    ui.selectable_value(&mut current_scale, 1.25, "125%");
                    ui.selectable_value(&mut current_scale, 1.50, "150%");
                });
            if current_scale != self.config.settings_scale {
                self.config.settings_scale = current_scale;
            }
            reset_btn(ui, (self.config.settings_scale - 1.0).abs() > 1e-3, || self.config.settings_scale = 1.0);
        });
        checkbox_reset(ui, &mut self.config.open_settings_on_launch, "Open Settings on Launch", DEFAULT_OPEN_SETTINGS_ON_LAUNCH);
        checkbox_reset(ui, &mut self.config.show_help_text, "Show Inline Help Text", DEFAULT_SHOW_HELP_TEXT);
    }

    // ── Performance tab ──────────────────────────────────────

    fn tab_performance(&mut self, ui: &mut egui::Ui) {
        section_header(ui, "Display");
        checkbox_reset(ui, &mut self.config.show_fps, "Show FPS Counter", false);
        ui.label(format!("Current FPS: {:.1}", self.current_fps));
        ui.add_space(6.0);
        ui.add(egui::Slider::new(&mut self.config.max_fps, 15..=60).text("Max FPS (lower = less GPU/DWM load)"));
        if self.config.max_fps != 60 && ui.small_button("Reset to 60").clicked() {
            self.config.max_fps = 60;
        }
        ui.add_space(6.0);
        
    }

    // ── Static helpers reused across tabs ────────────────────

    fn draw_dynamic_palette_settings(ui: &mut egui::Ui, id_suffix: &str, palette_size: &mut u8, variants: &mut u8, priority: &mut PalettePriority, show_size: bool, show_variants: bool) {
        ui.push_id(id_suffix, |ui| {
            ui.horizontal(|ui| {
                ui.label("Priority:");
                egui::ComboBox::from_id_source("priority")
                    .selected_text(priority.display_name())
                    .show_ui(ui, |ui| {
                        for p in PalettePriority::all() {
                            ui.selectable_value(priority, p, p.display_name());
                        }
                    });
            });
            if show_size {
                let mut v = *palette_size as i32;
                let r = ui.add(egui::Slider::new(&mut v, 1..=3).text("Colors from Cover"));
                if r.secondary_clicked() { v = 1; }
                *palette_size = v.clamp(1, 3) as u8;
            }
            if show_variants {
                let mut v = *variants as i32;
                let r = ui.add(egui::Slider::new(&mut v, 0..=5).text("Shade Variants per Color"));
                if r.secondary_clicked() { v = 0; }
                *variants = v.clamp(0, 5) as u8;
            }
        });
    }

    fn draw_palette_preview(ui: &mut egui::Ui, colors: &[[u8; 3]]) {
        ui.label("Current palette:");
        ui.horizontal_wrapped(|ui| {
            for c in colors {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 4.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
            }
        });
    }
}

fn fft_size_combo(ui: &mut egui::Ui, a: &mut visualizer_core::config::AudioSettings) {
    ui.horizontal(|ui| {
        ui.label("FFT Size:");
        egui::ComboBox::from_id_source("fft_size")
            .selected_text(format!("{}", a.fft_size))
            .show_ui(ui, |ui| {
                for size in [512usize, 1024, 2048, 4096, 8192] {
                    ui.selectable_value(&mut a.fft_size, size, format!("{size}"));
                }
            });
        reset_btn(ui, a.fft_size != 2048, || a.fft_size = 2048);
    });
}

// ═══════════════════════════════════════════════════════════════
// eframe::App
// ═══════════════════════════════════════════════════════════════

/// Body of the deferred settings viewport. Runs ONLY when the settings window
/// receives input / needs a repaint — it costs nothing while you aren't
/// interacting with it, and nothing at all while it is closed.
fn run_settings_viewport(
    ctx: &egui::Context,
    shared: &Mutex<SettingsState>,
    open: &AtomicBool,
    generation: &AtomicU64,
) {
    let mut st = lock_state(shared);

    // Window "X" button.
    if ctx.input(|i| i.viewport().close_requested()) {
        st.config.save();
        drop(st);
        open.store(false, Ordering::Relaxed);
        ctx.request_repaint_of(egui::ViewportId::ROOT);
        return;
    }

    // UI scale slider → resize the window.
    if (st.config.settings_scale - st.old_settings_scale).abs() > 0.01 {
        st.old_settings_scale = st.config.settings_scale;
        let scale = st.config.settings_scale.clamp(0.75, 1.5);
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(560.0 * scale, 720.0 * scale)));
    }

    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(egui::Color32::BLACK))
        .show(ctx, |ui| {
            st.draw_settings_ui(ui);
        });

    if let Some(idx) = st.pending_audio_preset_delete {
        egui::Window::new("Confirm Delete")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                let name = st.config.custom_audio_presets.get(idx)
                    .map(|p| p.name.clone())
                    .unwrap_or_default();
                ui.label(format!("Delete preset '{}'?", name));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Yes").clicked() {
                        if idx < st.config.custom_audio_presets.len() {
                            st.config.custom_audio_presets.remove(idx);
                        }
                        st.pending_audio_preset_delete = None;
                    }
                    if ui.button("No").clicked() {
                        st.pending_audio_preset_delete = None;
                    }
                });
            });
    }

     // Did this frame change anything the overlay cares about?
    // Serialising the whole config is not free, so only do it when the user
    // actually interacted (not on plain mouse-move frames), plus a slow
    // safety re-check in case something changed without input.
    let interacted = ctx.input(|i| {
        i.pointer.any_down()
            || i.pointer.any_released()
            || i.events.iter().any(|e| !matches!(
                e,
                egui::Event::PointerMoved(_) | egui::Event::PointerGone | egui::Event::WindowFocused(_)
            ))
    });
    let due = st.last_fp_check.elapsed() > std::time::Duration::from_millis(300);
    let mut changed = st.audio_dirty;
    if interacted || due || st.audio_dirty {
        let fp = config_fingerprint(&st.config);
        if fp != st.last_fingerprint {
            changed = true;
            st.last_fingerprint = fp;
        }
        st.last_fp_check = Instant::now();
    }
    
    let wants_close = st.close_requested;
    st.close_requested = false;
    drop(st);

    if changed {
        generation.fetch_add(1, Ordering::Release);
        ctx.request_repaint_of(egui::ViewportId::ROOT);
    }
    if wants_close {
        open.store(false, Ordering::Relaxed);
        ctx.request_repaint_of(egui::ViewportId::ROOT);
    }
}

impl eframe::App for VisualizerApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] { egui::Rgba::TRANSPARENT.to_array() }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        // Ignore the long gap after waking from idle so the FPS counter stays sane.
        if dt > 0.0 && dt < 0.1 {
            let instant_fps = 1.0 / dt;
            self.current_fps = if self.current_fps <= 0.0 {
                instant_fps
            } else {
                self.current_fps * 0.9 + instant_fps * 0.1
            };
        }
        if ctx.pixels_per_point() != 1.0 { ctx.set_pixels_per_point(1.0); }

        // ── Commands from tray / hotkey threads ───────────────
        while let Ok(cmd) = self.ui_cmd_rx.try_recv() {
            match cmd {
                UiCmd::OpenSettings => self.settings_open.store(true, Ordering::Relaxed),
                UiCmd::ToggleSettings => {
                    let v = self.settings_open.load(Ordering::Relaxed);
                    self.settings_open.store(!v, Ordering::Relaxed);
                }
                UiCmd::Quit => self.should_quit.store(true, Ordering::Relaxed),
            }
        }
        if self.should_quit.load(Ordering::Relaxed) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        // ── Pick up config edits made in the settings window ──
        let generation = self.config_gen.load(Ordering::Acquire);
        if generation != self.seen_gen {
            // try_lock: never stall the overlay while the settings UI is mid-frame.
            if let Some(mut st) = try_lock_state(&self.shared) {
                self.seen_gen = generation;
                self.config = st.config.clone();
                if st.audio_dirty || self.config.audio != *self.audio_settings.load_full() {
                    self.audio_settings.store(Arc::new(self.config.audio.clone()));
                    st.audio_dirty = false;
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(4));
            }
        }

        self.poll_track_updates(ctx);

        let new_hash = self.palette_fingerprint();
        if new_hash != self.last_palette_hash {
            self.update_palettes();
            self.last_palette_hash = new_hash;
            // Mirror to the settings window (used for its colour previews).
            if let Some(mut st) = try_lock_state(&self.shared) {
                st.active_palettes = self.active_palettes.clone();
                self.palette_mirrored = true;
            } else {
                self.palette_mirrored = false;
                ctx.request_repaint_after(std::time::Duration::from_millis(4));
            }
            if self.settings_open.load(Ordering::Relaxed) {
                ctx.request_repaint_of(egui::ViewportId::from_hash_of("settings"));
            }
        }

        if !self.palette_mirrored {
            if let Some(mut st) = try_lock_state(&self.shared) {
                st.active_palettes = self.active_palettes.clone();
                self.palette_mirrored = true;
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(4));
            }
        }

        if self.config.track_info.font_path != self.loaded_font_path {
            self.reload_custom_font(ctx);
        }

        // ── Bar buffer resize ─────────────────────────────────
        let n_bars = self.config.audio.num_bars;
        if self.current_bars.len() != n_bars {
            self.current_bars.resize(n_bars, 0.0);
            self.current_peaks.resize(n_bars, 0.0);
        }

        // ── Receive latest audio frame ────────────────────────
        let mut latest: Option<BarFrame> = None;
        while let Ok(frame) = self.audio_rx.try_recv() { latest = Some(frame); }
        if let Some(frame) = latest {
            let limit = self.current_bars.len().min(frame.values.len());
            self.current_bars[..limit].copy_from_slice(&frame.values[..limit]);
            self.current_peaks[..limit].copy_from_slice(&frame.peaks[..limit]);
            self.last_bpm = frame.bpm;
        }

        // ── Mouse passthrough (set once) ──────────────────────
        if !self.is_passthrough_set {
            ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(true));
            self.is_passthrough_set = true;
        }

        // ── Overlay drawing — direct painter, no Area overhead ──
        let rect = ctx.screen_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("overlay"),
        ));

        if self.config.bg_opacity > 0 {
            let c = self.config.bg_color;
            let bg_color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], self.config.bg_opacity);
            painter.rect_filled(rect, 0.0, bg_color);
        }

        self.draw_bars(&painter, ctx, rect);
        self.draw_track_info(&painter, ctx, rect);

        if self.config.show_fps {
            painter.text(
                rect.left_top() + egui::vec2(20.0, 20.0),
                egui::Align2::LEFT_TOP,
                format!("FPS: {:.1}", self.current_fps),
                egui::FontId::monospace(16.0),
                egui::Color32::GREEN,
            );
        }

        // ── Settings viewport (deferred: repaints independently) ──
        let open = self.settings_open.load(Ordering::Relaxed);
        let settings_id = egui::ViewportId::from_hash_of("settings");
        if open {
            let scale = self.config.settings_scale.clamp(0.75, 1.5);
            let mut builder = egui::ViewportBuilder::default()
                .with_title("Rust Visualizer — Settings")
                .with_inner_size([560.0 * scale, 720.0 * scale])
                .with_decorations(true)
                .with_resizable(true)
                .with_always_on_top();
            if self.settings_icon.width > 0 {
                builder = builder.with_icon(self.settings_icon.clone());
            }

            // Keep the mirrors fresh (try_lock: never block the overlay).
            if let Ok(mut st) = self.shared.try_lock() {
                st.last_bpm = self.last_bpm;
                st.current_fps = self.current_fps;
            }

            let shared = self.shared.clone();
            let open_flag = self.settings_open.clone();
            let generation = self.config_gen.clone();
            ctx.show_viewport_deferred(settings_id, builder, move |ctx, _class| {
                run_settings_viewport(ctx, &shared, &open_flag, &generation);
            });
        } else if self.was_settings_open {
            ctx.send_viewport_cmd_to(settings_id, egui::ViewportCommand::Close);
        }
        self.was_settings_open = open;

        // ── Repaint policy ────────────────────────────────────
        // 60 FPS (VSync-paced) ONLY while something is visibly moving. When
        // the bars have fully decayed we stop requesting frames, so the
        // overlay costs ~0 % CPU/GPU and DWM stops recompositing it.
        // Wake-ups come from: audio frames, track changes, tray/hotkey
        // events and settings edits (all call request_repaint themselves).
        let moving = self.current_bars.iter().any(|&v| v > 0.0005)
            || self.current_peaks.iter().any(|&v| v > 0.0005);
        if moving {
            // While animating, the overlay drives its own repaints so the FPS
            // cap is honoured; audio threads stay quiet (see `ui_active`).
            self.ui_active.store(true, Ordering::Release);
            let fps = self.config.max_fps.clamp(15, 60);
            if fps >= 60 {
                // VSync (60 Hz) paces this.
                ctx.request_repaint_of(egui::ViewportId::ROOT);
            } else {
                let target = std::time::Duration::from_secs_f32(1.0 / fps as f32);
                // This closure runs inside the ROOT pass, so this targets ROOT.
                ctx.request_repaint_after(target.saturating_sub(now.elapsed()));
            }
        } else {
            self.ui_active.store(false, Ordering::Release);
            // A frame may have arrived between draining the channel and the
            // store above; make sure it isn't left unseen.
            if !self.audio_rx.is_empty() {
                wake_root(ctx);
            }
        }
    }

    fn on_exit(&mut self, _gl: Option<&glow::Context>) { self.config.save(); }
}

// ═══════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════

fn main() -> eframe::Result<()> {
    let (egui_icon, _) = load_icon();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_transparent(true)
            .with_decorations(false)
            .with_always_on_top()
            .with_maximized(true)
            .with_icon(egui_icon),
        vsync: true,
        ..Default::default()
    };
    eframe::run_native(
        "Rust Visualizer",
        options,
        Box::new(|cc| Box::new(VisualizerApp::new(cc.egui_ctx.clone()))),
    )
}
