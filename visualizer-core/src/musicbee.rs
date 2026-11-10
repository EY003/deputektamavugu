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
// Icon loader (unchanged)
// ═══════════════════════════════════════════════════════════════

fn load_icon() -> (egui::IconData, tray_icon::Icon) {
    let icon_bytes = include_bytes!("icon.png");
    let image = image::load_from_memory(icon_bytes).unwrap_or_else(|_| {
        image::DynamicImage::new_rgba8(1, 1)
    });
    let rgba = image.to_rgba8();
    let (w, h) = rgba.dimensions();
    let raw = rgba.into_raw();
    let egui_icon = egui::IconData { rgba: raw.clone(), width: w, height: h };
    let tray = tray_icon::Icon::from_rgba(raw, w, h).unwrap();
    (egui_icon, tray)
}

// ═══════════════════════════════════════════════════════════════
// Utilities (mostly unchanged)
// ═══════════════════════════════════════════════════════════════

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
// Fonts (unchanged)
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
// Palette cache (unchanged)
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
// Fingerprint helpers (unchanged)
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
// Settings UI helpers (unchanged)
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
// Settings tabs enum (unchanged)
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
// App struct — now includes overlay rect and mesh cache
// ═══════════════════════════════════════════════════════════════

enum UiCmd {
    OpenSettings,
    ToggleSettings,
    Quit,
}

struct SettingsState {
    config: AppConfig,
    audio_dirty: bool,
    close_requested: bool,
    last_fingerprint: u64,
    old_settings_scale: f32,
    new_preset_name: String,
    new_audio_preset_name: String,
    last_decay_mode: DecayMode,
    settings_tab: SettingsTab,
    color_subtab: ColorSubTab,
    settings_search: String,
    pending_audio_preset_delete: Option<usize>,
    active_palettes: ActivePalettes,
    last_bpm: f32,
    current_fps: f32,
    system_fonts: Vec<(String, String)>,
}

fn lock_state(m: &Mutex<SettingsState>) -> std::sync::MutexGuard<'_, SettingsState> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn config_fingerprint(c: &AppConfig) -> u64 {
    use std::hash::{Hash, Hasher};
    match toml::to_string(c) {
        Ok(s) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }
        Err(_) => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0),
    }
}

/// Cached mesh for the bar geometry. We reuse the vertex/index buffers
/// across frames instead of rebuilding them from scratch.
struct BarMeshCache {
    mesh: egui::Mesh,
    last_num_bars: usize,
    last_bar_width: f32,
    last_gap: f32,
    last_segments: usize,
}

impl Default for BarMeshCache {
    fn default() -> Self {
        Self {
            mesh: egui::Mesh::default(),
            last_num_bars: 0,
            last_bar_width: 0.0,
            last_gap: 0.0,
            last_segments: 0,
        }
    }
}

impl BarMeshCache {
    fn clear(&mut self) {
        self.mesh.vertices.clear();
        self.mesh.indices.clear();
    }

    fn reserve(&mut self, verts: usize, indices: usize) {
        if self.mesh.vertices.capacity() < verts {
            self.mesh.vertices.reserve(verts - self.mesh.vertices.len());
        }
        if self.mesh.indices.capacity() < indices {
            self.mesh.indices.reserve(indices - self.mesh.indices.len());
        }
    }
}

struct VisualizerApp {
    config: AppConfig,
    current_bars: Vec<f32>,
    current_peaks: Vec<f32>,
    last_bpm: f32,
    audio_rx: crossbeam_channel::Receiver<BarFrame>,
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

    shared: Arc<Mutex<SettingsState>>,
    settings_open: Arc<AtomicBool>,
    config_gen: Arc<AtomicU64>,
    seen_gen: u64,
    was_settings_open: bool,
    settings_icon: egui::IconData,
    ui_cmd_rx: crossbeam_channel::Receiver<UiCmd>,

    _tray_icon: TrayIcon,
    _hotkey_manager: GlobalHotKeyManager,

    // ── New: overlay geometry + mesh cache ───────────────────
    overlay_rect: egui::Rect,
    bar_mesh_cache: BarMeshCache,
}
