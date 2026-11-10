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
// Icon loader
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
// Small utilities
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
    if hard { if pos < boundary { 0.0 } else { 1.0 } }
    else { smoothstep(boundary - softness, boundary + softness, pos) }
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
    let h = rect.height(); let w = rect.width();
    match dir {
        VisualizerDirection::Up => egui::Rect::from_min_max(
            egui::pos2(rect.left(), rect.bottom() - h * t1),
            egui::pos2(rect.right(), rect.bottom() - h * t0)),
        VisualizerDirection::Down => egui::Rect::from_min_max(
            egui::pos2(rect.left(), rect.top() + h * t0),
            egui::pos2(rect.right(), rect.top() + h * t1)),
        VisualizerDirection::Right => egui::Rect::from_min_max(
            egui::pos2(rect.left() + w * t0, rect.top()),
            egui::pos2(rect.left() + w * t1, rect.bottom())),
        VisualizerDirection::Left => egui::Rect::from_min_max(
            egui::pos2(rect.right() - w * t1, rect.top()),
            egui::pos2(rect.right() - w * t0, rect.bottom())),
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
    painter: &egui::Painter, dir: VisualizerDirection, anchor: f32, perp: f32,
    peak_len: f32, bar_width: f32, color: egui::Color32, is_smooth: bool,
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
struct MotionSample { phase_a: f32, phase_b: f32, blend_t: f32, intensity: f32 }
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
                        result.push((name.to_string_lossy().to_string(), path.to_string_lossy().to_string()));
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
            1u8.hash(h); palette_size.hash(h); variants_per_color.hash(h); priority.hash(h);
        }
    }
}
fn hash_gradient_source<H: std::hash::Hasher>(s: &GradientSource, h: &mut H) {
    use std::hash::Hash;
    match s {
        GradientSource::Static(stops) => {
            0u8.hash(h);
            for stop in stops { stop.color.hash(h); stop.position.to_bits().hash(h); }
        }
        GradientSource::Dynamic { palette_size, variants_per_color, priority } => {
            1u8.hash(h); palette_size.hash(h); variants_per_color.hash(h); priority.hash(h);
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
    ui.add_space(2.0); ui.separator(); ui.add_space(4.0);
}

fn reset_btn<F: FnOnce()>(ui: &mut egui::Ui, changed: bool, on_reset: F) {
    let text = egui::RichText::new("Reset").size(12.0);
    if ui.add_enabled(changed, egui::Button::new(text).small()).clicked() { on_reset(); }
}

fn help_text(ui: &mut egui::Ui, show: bool, text: &str) {
    if show {
        ui.add_space(2.0);
        ui.label(egui::RichText::new(format!("ℹ {}", text)).size(12.0)
            .color(egui::Color32::from_gray(150)).italics());
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
    ui: &mut egui::Ui, value: &mut T, default: T, label: &str, id: &str,
    width: f32, options: &[T], name: fn(T) -> &'static str,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_source(id).selected_text(name(*value)).width(width)
            .show_ui(ui, |ui| {
                for &opt in options { ui.selectable_value(value, opt, name(opt)); }
            });
        reset_btn(ui, *value != default, || *value = default);
    });
}

// ═══════════════════════════════════════════════════════════════
// Settings enums
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab { Visualizer, Colors, Audio, TrackInfo, Appearance, Performance }
#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorSubTab { Single, Multi, Gradient, Animated }

const SEARCH_INDEX: &[(SettingsTab, &str, &[&str])] = &[
    (SettingsTab::Visualizer, "Direction", &["direction","up","down","left","right"]),
    (SettingsTab::Visualizer, "Frequency Layout", &["frequency","mirrored","layout"]),
    (SettingsTab::Visualizer, "Bar Shape", &["bar","shape","step","smooth","interpolation"]),
    (SettingsTab::Visualizer, "Bars & Position", &["bar","width","gap","height","offset"]),
    (SettingsTab::Visualizer, "Outlines", &["outline","border"]),
    (SettingsTab::Colors, "Color Source", &["color","palette","album","dynamic"]),
    (SettingsTab::Colors, "Distribution", &["blocks","sequential","random","vertical"]),
    (SettingsTab::Colors, "Gradient", &["gradient","stops","angle"]),
    (SettingsTab::Colors, "Motion", &["motion","animated","shape"]),
    (SettingsTab::Audio, "Presets", &["preset","audio","custom"]),
    (SettingsTab::Audio, "Signal Processing", &["fft","window","channel","weighting"]),
    (SettingsTab::Audio, "Amplitude", &["sensitivity","gain","gamma","noise","agc"]),
    (SettingsTab::Audio, "Equalizer", &["eq","bass","mid","treble","tilt"]),
    (SettingsTab::Audio, "Ballistics", &["attack","decay","gravity","peak"]),
    (SettingsTab::Audio, "Rhythm", &["beat","bpm","tempo","phase"]),
    (SettingsTab::TrackInfo, "Cover Art", &["cover","art","album"]),
    (SettingsTab::TrackInfo, "Text", &["title","artist","font","text"]),
    (SettingsTab::TrackInfo, "Shadow", &["shadow","blur"]),
    (SettingsTab::TrackInfo, "Outline", &["outline","border"]),
    (SettingsTab::Appearance, "Background", &["background","bg","opacity"]),
    (SettingsTab::Appearance, "UI", &["scale","size","help","launch"]),
    (SettingsTab::Performance, "Performance", &["fps","vsync","frame"]),
];

// ═══════════════════════════════════════════════════════════════
// Mesh builder — one draw call per frame for all bars.
// ═══════════════════════════════════════════════════════════════

/// Reusable mesh builder. Vertices and indices are kept alive between
/// frames so we only pay allocation the first time a larger buffer is
/// needed.
struct MeshBuilder {
    vertices: Vec<egui::epaint::Vertex>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn new() -> Self {
        Self { vertices: Vec::with_capacity(8192), indices: Vec::with_capacity(12288) }
    }

    fn reset(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    fn push_quad(&mut self, p1: egui::Pos2, p2: egui::Pos2, p3: egui::Pos2, p4: egui::Pos2, color: egui::Color32) {
        let base = self.vertices.len() as u32;
        let v = egui::epaint::Vertex { pos: p1, uv: egui::Pos2::ZERO, color };
        self.vertices.push(v);
        let v = egui::epaint::Vertex { pos: p2, uv: egui::Pos2::ZERO, color };
        self.vertices.push(v);
        let v = egui::epaint::Vertex { pos: p3, uv: egui::Pos2::ZERO, color };
        self.vertices.push(v);
        let v = egui::epaint::Vertex { pos: p4, uv: egui::Pos2::ZERO, color };
        self.vertices.push(v);
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Push a quad with per-corner colors (used by smooth mode).
    fn push_quad_colored(&mut self,
        p1: egui::Pos2, c1: egui::Color32,
        p2: egui::Pos2, c2: egui::Color32,
        p3: egui::Pos2, c3: egui::Color32,
        p4: egui::Pos2, c4: egui::Color32,
    ) {
        let base = self.vertices.len() as u32;
        self.vertices.push(egui::epaint::Vertex { pos: p1, uv: egui::Pos2::ZERO, color: c1 });
        self.vertices.push(egui::epaint::Vertex { pos: p2, uv: egui::Pos2::ZERO, color: c2 });
        self.vertices.push(egui::epaint::Vertex { pos: p3, uv: egui::Pos2::ZERO, color: c3 });
        self.vertices.push(egui::epaint::Vertex { pos: p4, uv: egui::Pos2::ZERO, color: c4 });
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Produce an `egui::Mesh` suitable for `painter.add(Shape::mesh(..))`.
    /// We hand out a clone so the internal buffers stay alive for reuse;
    /// the clone is a single memcpy per frame (cheaper than allocation
    /// churn).
    fn take_mesh(&self) -> egui::Mesh {
        egui::Mesh {
            indices: self.indices.clone(),
            vertices: self.vertices.clone(),
            texture_id: egui::TextureId::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Settings state (unchanged from original)
// ═══════════════════════════════════════════════════════════════

enum UiCmd { OpenSettings, ToggleSettings, Quit }

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
            s.hash(&mut h); h.finish()
        }
        Err(_) => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64).unwrap_or(0),
    }
}

// ═══════════════════════════════════════════════════════════════
// VisualizerApp
// ═══════════════════════════════════════════════════════════════

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

    // ── Overlay geometry & mesh cache ────────────────────────
    /// The monitor rect the app was originally launched on. Captured once
    /// on the first frame while the window is still fullscreen.
    monitor_rect: Option<egui::Rect>,
    /// The rect (in monitor coordinates) the overlay is currently sized to.
    /// We resize the window to this on change.
    overlay_rect: egui::Rect,
    /// One reusable mesh builder for all bars.
    mesh: MeshBuilder,
}

impl VisualizerApp {
    fn new(ctx: egui::Context) -> Self {
        let config = AppConfig::load();

        let audio_ctx = ctx.clone();
        let engine = AudioEngine::new(config.audio.clone(), move || audio_ctx.request_repaint());
        let should_quit = Arc::new(AtomicBool::new(false));
        musicbee::log_smtc("APP: Initializing VisualizerApp.");
        let audio_settings = engine.settings.clone();

        let (track_tx, track_rx) = crossbeam_channel::unbounded();
        let smtc_session_count = Arc::new(AtomicUsize::new(0));
        let smtc_ctx = ctx.clone();
        musicbee::spawn_smtc_listener(
            track_tx, should_quit.clone(), smtc_session_count.clone(),
            move || smtc_ctx.request_repaint(),
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
            .build().unwrap();

        let hotkey_manager = GlobalHotKeyManager::new().unwrap();
        let hotkey = HotKey::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyV);
        hotkey_manager.register(hotkey).unwrap();

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
                    if let Some(cmd) = cmd { let _ = tx.send(cmd); wake.request_repaint(); }
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
                        wake.request_repaint();
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
            monitor_rect: None,
            overlay_rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 100.0)),
            mesh: MeshBuilder::new(),
        }
    }

    fn visual_num_bars(&self) -> usize {
        let base = self.current_bars.len();
        if self.config.frequency_direction == FrequencyDirection::Mirrored { base * 2 } else { base }
    }

    // ── Overlay geometry ─────────────────────────────────────

    /// Compute the tight bounding rect of everything we draw, expressed in
    /// monitor coordinates. This is what the overlay window is resized to.
    fn compute_overlay_rect(&self, monitor: egui::Rect) -> egui::Rect {
        let n_bars = self.visual_num_bars().max(1);
        let stride = self.config.bar_width + self.config.bar_gap;
        let bars_extent = (n_bars as f32 * stride - self.config.bar_gap).max(1.0);

        let scale = if self.config.max_bar_height > 0.0 {
            self.config.max_bar_height
        } else {
            monitor.height() * self.config.height_scale
        };
        let scale = scale.max(1.0);

        let cx = monitor.center().x + self.config.x_offset;
        let cy = monitor.center().y + self.config.y_offset;

        let bars_rect = match self.config.visualizer_direction {
            VisualizerDirection::Up => egui::Rect::from_min_max(
                egui::pos2(cx - bars_extent / 2.0, monitor.bottom() + self.config.y_offset - scale),
                egui::pos2(cx + bars_extent / 2.0, monitor.bottom() + self.config.y_offset),
            ),
            VisualizerDirection::Down => egui::Rect::from_min_max(
                egui::pos2(cx - bars_extent / 2.0, monitor.top() + self.config.y_offset),
                egui::pos2(cx + bars_extent / 2.0, monitor.top() + self.config.y_offset + scale),
            ),
            VisualizerDirection::Right => egui::Rect::from_min_max(
                egui::pos2(monitor.left() + self.config.x_offset, cy - bars_extent / 2.0),
                egui::pos2(monitor.left() + self.config.x_offset + scale, cy + bars_extent / 2.0),
            ),
            VisualizerDirection::Left => egui::Rect::from_min_max(
                egui::pos2(monitor.right() + self.config.x_offset - scale, cy - bars_extent / 2.0),
                egui::pos2(monitor.right() + self.config.x_offset, cy + bars_extent / 2.0),
            ),
        };

        let mut bbox = bars_rect;

        // Track info bbox — same positioning math as drawing.
        let cfg = &self.config.track_info;
        let info_present = (cfg.show_cover_art && self.cover_texture.is_some())
            || cfg.show_title || cfg.show_artist;
        if info_present {
            let margin = 40.0;
            let art_size = if cfg.show_cover_art { cfg.cover_size } else { 0.0 };
            let est_text_h = cfg.title_font_size * cfg.line_height
                + cfg.artist_font_size * cfg.line_height
                + 8.0;
            let est_text_w = 500.0_f32;

            let art_x = monitor.right() - margin - art_size + cfg.cover_x_offset;
            let art_y = monitor.top() + margin + cfg.cover_y_offset;

            let info_left = if art_size > 0.0 { art_x - est_text_w } else { art_x - est_text_w };
            let info_right = (art_x + art_size).max(art_x + est_text_w.min(art_size));
            let info_top = art_y;
            let info_bottom = art_y + art_size + est_text_h;

            let info_rect = egui::Rect::from_min_max(
                egui::pos2(info_left, info_top),
                egui::pos2(info_right, info_bottom),
            );
            bbox = bbox.union(info_rect);
        }

        // FPS badge
        if self.config.show_fps {
            let fps_rect = egui::Rect::from_min_size(monitor.left_top() + egui::vec2(20.0, 20.0), egui::vec2(160.0, 30.0));
            bbox = bbox.union(fps_rect);
        }

        // Pad for outlines / shadows / peak-hold markers / rounding.
        let pad = 32.0;
        let bbox = bbox.expand(pad);

        // Ensure a minimum size so the window never collapses to nothing.
        let min = egui::vec2(80.0, 60.0);
        let size = bbox.size().max(min);
        let center = bbox.center();
        let mut result = egui::Rect::from_center_size(center, size);

        // Clamp to monitor
        result = result.intersect(monitor);

        result
    }

    // ── Palette updates ──────────────────────────────────────

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
                    .map(|s| s.to_vec()).unwrap_or_else(|| {
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
                    .map(|s| s.to_vec()).unwrap_or_else(|| {
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
                    .map(|s| s.to_vec()).unwrap_or_else(|| {
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

    // ── Track updates ────────────────────────────────────────

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

    // ── Colors ───────────────────────────────────────────────

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
                        let bpc = self.config.sequential_bars_per_color.max(1);
                        (i / bpc) % num_colors
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
                let mut min_proj = f32::MAX; let mut max_proj = f32::MIN;
                for c in corners {
                    let p = c.x * dir.x + c.y * dir.y;
                    min_proj = min_proj.min(p); max_proj = max_proj.max(p);
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
                let b = tri01(base_t * 0.5); let t = wipe_blend(x, b, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::VerticalWipe => {
                let b = tri01(base_t * 0.5); let t = wipe_blend(y, b, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::DiagonalWipe => {
                let b = tri01(base_t * 0.5); let diag = (x + y) * 0.5;
                let t = wipe_blend(diag, b, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::RadialWipe => {
                let b = tri01(base_t * 0.5);
                let dx = x - 0.5; let dy = y - 0.5;
                let r = (dx * dx + dy * dy).sqrt() * 2.0;
                let t = wipe_blend(r, b, m.wipe_softness, hard);
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::Spiral => {
                let dx = x - 0.5; let dy = y - 0.5;
                let r = (dx * dx + dy * dy).sqrt() * 2.0;
                let angle = dy.atan2(dx) / TAU + 0.5;
                let t = (angle + r * 2.0 + base_t).fract();
                MotionSample { phase_a: base_t, phase_b: base_t + 0.5, blend_t: t, intensity: 1.0 }
            }
            MotionShape::WaveInterference => {
                let w1 = (x * freq * TAU + base_t * TAU).sin();
                let w2 = (y * freq * TAU + base_t * TAU * 0.7).sin();
                let wave = (w1 + w2) * 0.25 + 0.5;
                MotionSample { phase_a: base_t + wave * 0.5, intensity: 1.0, ..Default::default() }
            }
            MotionShape::Ripple => {
                let dx = x - 0.5; let dy = y - 0.5;
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

    // ── Track info drawing ───────────────────────────────────

    fn draw_styled_text(&self, ctx: &egui::Context, painter: &egui::Painter, text: &str,
        pos: egui::Pos2, font_id: &egui::FontId, align: egui::Align2,
        color: egui::Color32, cfg: &TrackInfoConfig)
    {
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

    fn draw_track_info(&self, painter: &egui::Painter, ctx: &egui::Context, _local_rect: egui::Rect, overlay_origin: egui::Pos2, monitor: egui::Rect) {
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
        // Positions computed in *monitor* coordinates, then translated into
        // the overlay window's local coordinates.
        let art_x_m = monitor.right() - margin - art_size + cfg.cover_x_offset;
        let art_y_m = monitor.top() + margin + cfg.cover_y_offset;
        if cover_visible {
            if let Some(tex) = &self.cover_texture {
                let art_rect = egui::Rect::from_min_size(
                    egui::pos2(art_x_m, art_y_m) - overlay_origin.to_vec2(),
                    egui::vec2(art_size, art_size),
                );
                let tint = egui::Color32::from_white_alpha(cfg.cover_opacity);
                painter.image(tex.id(), art_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), tint);
            }
        }
        let mut text_y_m = monitor.top() + margin + cfg.text_y_offset;
        if cover_visible { text_y_m += art_size + text_gap; }
        let text_x_m = monitor.right() - margin + cfg.text_x_offset;
        let mut current_y_m = text_y_m;
        if cfg.show_artist && !info.artist.is_empty() {
            let pos = egui::pos2(text_x_m, current_y_m) - overlay_origin.to_vec2();
            self.draw_styled_text(ctx, painter, &info.artist, pos, &artist_font, align, text_color, cfg);
            current_y_m += cfg.artist_font_size * cfg.line_height;
        }
        if cfg.show_title && !info.title.is_empty() {
            let pos = egui::pos2(text_x_m, current_y_m) - overlay_origin.to_vec2();
            self.draw_styled_text(ctx, painter, &info.title, pos, &title_font, align, text_color, cfg);
        }
    }

    // ── Bars ─────────────────────────────────────────────────

    /// Draw all bars with a single `painter.add(mesh)` call. Both `Step`
    /// and `Smooth` modes, all directions, all color modes and vertical
    /// segmentation are handled here so the whole bar region is one draw
    /// call instead of N or N*S.
    fn draw_bars(&mut self, painter: &egui::Painter, ctx: &egui::Context,
                 local_rect: egui::Rect, canvas_h: f32, canvas_w: f32)
    {
        let base_num_bars = self.current_bars.len();
        if base_num_bars == 0 { return; }
        let time = ctx.input(|i| i.time) as f32;
        let visual_num_bars = self.visual_num_bars();

        let bar_width = self.config.bar_width;
        let gap = self.config.bar_gap;
        let stride = bar_width + gap;
        let total_size = visual_num_bars as f32 * stride - gap;
        let max_h = self.config.max_bar_height;
        let radius = self.config.bar_radius;
        let freq_dir = self.config.frequency_direction;
        let dir = self.config.visualizer_direction;
        let is_smooth = self.config.bar_interpolation == BarInterpolation::Smooth;
        let outline_on = self.config.visualizer_outline_enabled && self.config.visualizer_outline_thickness > 0.0;

        // Choose the coordinate scale: bars extend along the axis dictated
        // by direction. We use monitor size as the canvas so the total
        // extent is stable regardless of the (tiny) window we draw into.
        let (rect_len, start_pos, anchor) = match dir {
            VisualizerDirection::Up => (
                canvas_h * self.config.height_scale,
                local_rect.center().x - total_size / 2.0,
                local_rect.bottom(),
            ),
            VisualizerDirection::Down => (
                canvas_h * self.config.height_scale,
                local_rect.center().x - total_size / 2.0,
                local_rect.top(),
            ),
            VisualizerDirection::Right => (
                canvas_w * self.config.height_scale,
                local_rect.center().y - total_size / 2.0,
                local_rect.left(),
            ),
            VisualizerDirection::Left => (
                canvas_w * self.config.height_scale,
                local_rect.center().y - total_size / 2.0,
                local_rect.right(),
            ),
        };

        // Whether bars use per-vertical-segment coloring (the vertical
        // color modes and vertical motion shapes).
        let is_static_multi_vertical = self.config.color_mode == ColorMode::StaticMultiple
            && self.config.static_mode == StaticDistributionMode::Vertical;
        let is_animated_vertical = self.config.color_mode == ColorMode::Animated
            && self.config.color_change_motion.shape.uses_vertical_axis();
        let is_vertical = is_static_multi_vertical || is_animated_vertical;
        let n_segments = if is_vertical {
            if is_static_multi_vertical {
                self.active_palettes.static_multi_colors.len().clamp(1, 32)
            } else {
                self.active_palettes.animated_colors.len().max(1).clamp(6, 16)
            }
        } else { 1 };

        // Bounding rect used for gradient projection (matches original).
        let gradient_rect = match dir {
            VisualizerDirection::Up => egui::Rect::from_min_max(
                egui::pos2(local_rect.center().x - total_size / 2.0, local_rect.bottom() - rect_len.max(1.0)),
                egui::pos2(local_rect.center().x + total_size / 2.0, local_rect.bottom()),
            ),
            VisualizerDirection::Down => egui::Rect::from_min_max(
                egui::pos2(local_rect.center().x - total_size / 2.0, local_rect.top()),
                egui::pos2(local_rect.center().x + total_size / 2.0, local_rect.top() + rect_len.max(1.0)),
            ),
            VisualizerDirection::Right => egui::Rect::from_min_max(
                egui::pos2(local_rect.left(), local_rect.center().y - total_size / 2.0),
                egui::pos2(local_rect.left() + rect_len.max(1.0), local_rect.center().y + total_size / 2.0),
            ),
            VisualizerDirection::Left => egui::Rect::from_min_max(
                egui::pos2(local_rect.right() - rect_len.max(1.0), local_rect.center().y - total_size / 2.0),
                egui::pos2(local_rect.right(), local_rect.center().y + total_size / 2.0),
            ),
        };

        self.mesh.reset();

        if is_smooth {
            self.build_smooth_mesh(
                painter, &gradient_rect, dir, start_pos, anchor, rect_len,
                total_size, bar_width, stride, visual_num_bars, base_num_bars,
                freq_dir, max_h, n_segments, is_vertical, time,
            );
        } else {
            self.build_step_mesh(
                painter, &gradient_rect, dir, start_pos, anchor, rect_len,
                bar_width, stride, visual_num_bars, base_num_bars,
                freq_dir, max_h, radius, n_segments, is_vertical, time,
            );
        }

        // One draw call for all bars.
        if !self.mesh.indices.is_empty() {
            painter.add(egui::Shape::mesh(self.mesh.take_mesh()));
        }

        // Outlines (only when enabled) and peak-hold markers stay as
        // individual small painter calls — they're cheap.
        if outline_on {
            let oc = self.config.visualizer_outline_color;
            let stroke_color = self.apply_vis(egui::Color32::from_rgb(oc[0], oc[1], oc[2]));
            let stroke = egui::Stroke::new(self.config.visualizer_outline_thickness, stroke_color);
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let len = (self.current_bars[idx] * rect_len).min(if max_h > 0.0 { max_h } else { rect_len });
                if len <= 0.5 { continue; }
                let rect = self.bar_rect(dir, start_pos + i as f32 * stride, bar_width, anchor, len);
                painter.rect_stroke(rect, radius, stroke);
            }
        }

        if self.config.audio.peak_hold_ms > 0.0 {
            for i in 0..visual_num_bars {
                let idx = frequency_index(i, base_num_bars, freq_dir);
                let mut peak_len = self.current_peaks[idx] * rect_len;
                if max_h > 0.0 { peak_len = peak_len.min(max_h); }
                if peak_len <= 0.001 { continue; }
                let perp = match dir {
                    VisualizerDirection::Up | VisualizerDirection::Down => start_pos + i as f32 * stride + bar_width / 2.0,
                    VisualizerDirection::Left | VisualizerDirection::Right => start_pos + i as f32 * stride + bar_width / 2.0,
                };
                let t = (peak_len / rect_len).clamp(0.0, 1.0);
                let color = if is_vertical {
                    self.vertical_color(dir, perp, anchor, peak_len, t, &gradient_rect, time)
                } else {
                    self.get_color_at(i, visual_num_bars, perp, anchor, t, gradient_rect, time)
                };
                draw_peak_hold(painter, dir, anchor, perp, peak_len, bar_width, color, is_smooth);
            }
        }
    }

    #[inline]
    fn bar_rect(&self, dir: VisualizerDirection, perp_start: f32, bar_width: f32, anchor: f32, len: f32) -> egui::Rect {
        match dir {
            VisualizerDirection::Up => egui::Rect::from_min_max(
                egui::pos2(perp_start, anchor - len), egui::pos2(perp_start + bar_width, anchor)),
            VisualizerDirection::Down => egui::Rect::from_min_max(
                egui::pos2(perp_start, anchor), egui::pos2(perp_start + bar_width, anchor + len)),
            VisualizerDirection::Right => egui::Rect::from_min_max(
                egui::pos2(anchor, perp_start), egui::pos2(anchor + len, perp_start + bar_width)),
            VisualizerDirection::Left => egui::Rect::from_min_max(
                egui::pos2(anchor - len, perp_start), egui::pos2(anchor, perp_start + bar_width)),
        }
    }

    fn vertical_color(&self, dir: VisualizerDirection, _perp: f32, _anchor: f32, _len: f32, t: f32, gradient_rect: &egui::Rect, time: f32) -> egui::Color32 {
        match self.config.color_mode {
            ColorMode::StaticMultiple => {
                let blocks = self.config.vertical_color_style == VerticalColorStyle::Blocks;
                self.apply_vis(static_multi_vertical_color(&self.active_palettes.static_multi_colors, t, blocks))
            }
            ColorMode::Animated => self.apply_vis(self.sample_color_changing(0.5, t, time)),
            _ => {
                // Fallback: sample via get_color_at
                let _ = (dir, gradient_rect);
                self.apply_vis(egui::Color32::WHITE)
            }
        }
    }

    /// Build the step-mode mesh (rectangles) into `self.mesh`.
    #[allow(clippy::too_many_arguments)]
    fn build_step_mesh(&mut self, painter: &egui::Painter, gradient_rect: &egui::Rect,
        dir: VisualizerDirection, start_pos: f32, anchor: f32, rect_len: f32,
        bar_width: f32, stride: f32, visual_num_bars: usize, base_num_bars: usize,
        freq_dir: FrequencyDirection, max_h: f32, radius: f32, n_segments: usize,
        is_vertical: bool, time: f32)
    {
        let _ = painter;
        for i in 0..visual_num_bars {
            let idx = frequency_index(i, base_num_bars, freq_dir);
            let len = (self.current_bars[idx] * rect_len).min(if max_h > 0.0 { max_h } else { rect_len });
            if len <= 0.5 { continue; }
            let perp_start = start_pos + i as f32 * stride;
            let bar_rect = self.bar_rect(dir, perp_start, bar_width, anchor, len);

            if !is_vertical || n_segments <= 1 {
                // Single quad per bar.
                let color = self.get_color_at(i, visual_num_bars,
                    bar_rect.center().x, bar_rect.center().y, 0.5, *gradient_rect, time);
                // Manual quad to skip per-bar rect_filled.
                let (p1, p2, p3, p4) = match dir {
                    VisualizerDirection::Up | VisualizerDirection::Down => (
                        bar_rect.left_top(), bar_rect.left_bottom(),
                        bar_rect.right_bottom(), bar_rect.right_top()),
                    VisualizerDirection::Left | VisualizerDirection::Right => (
                        bar_rect.left_top(), bar_rect.right_top(),
                        bar_rect.right_bottom(), bar_rect.left_bottom()),
                };
                let _ = radius; // rounding handled by painter.rect_filled normally; keeping simple quads here
                self.mesh.push_quad(p1, p2, p3, p4, color);
            } else {
                // N vertical segments per bar.
                for k in 0..n_segments {
                    let t0 = k as f32 / n_segments as f32;
                    let t1 = (k + 1) as f32 / n_segments as f32;
                    let seg = segment_rect(bar_rect, dir, t0, t1);
                    let t_mid = (t0 + t1) * 0.5;
                    let color = match self.config.color_mode {
                        ColorMode::StaticMultiple => {
                            let blocks = self.config.vertical_color_style == VerticalColorStyle::Blocks;
                            self.apply_vis(static_multi_vertical_color(&self.active_palettes.static_multi_colors, t_mid, blocks))
                        }
                        ColorMode::Animated => {
                            let mid = seg.center();
                            let x_norm = ((mid.x - gradient_rect.left()) / gradient_rect.width().max(1.0)).clamp(0.0, 1.0);
                            self.apply_vis(self.sample_color_changing(x_norm, t_mid, time))
                        }
                        _ => self.get_color_at(i, visual_num_bars, seg.center().x, seg.center().y, t_mid, *gradient_rect, time),
                    };
                    let (p1, p2, p3, p4) = (seg.left_top(), seg.left_bottom(), seg.right_bottom(), seg.right_top());
                    self.mesh.push_quad(p1, p2, p3, p4, color);
                }
            }
        }
    }

    /// Build the smooth-mode mesh (interpolated curve) into `self.mesh`.
    #[allow(clippy::too_many_arguments)]
    fn build_smooth_mesh(&mut self, painter: &egui::Painter, gradient_rect: &egui::Rect,
        dir: VisualizerDirection, start_pos: f32, anchor: f32, rect_len: f32,
        total_size: f32, bar_width: f32, stride: f32, visual_num_bars: usize,
        base_num_bars: usize, freq_dir: FrequencyDirection, max_h: f32,
        n_segments: usize, is_vertical: bool, time: f32)
    {
        let _ = painter;
        // Precompute (perp, len, color_bot, color_top) per bar.
        struct BarSample { perp: f32, len: f32 }
        let mut bars: Vec<BarSample> = Vec::with_capacity(visual_num_bars);
        for i in 0..visual_num_bars {
            let idx = frequency_index(i, base_num_bars, freq_dir);
            let len = (self.current_bars[idx] * rect_len).min(if max_h > 0.0 { max_h } else { rect_len });
            let perp = start_pos + i as f32 * stride + bar_width / 2.0;
            bars.push(BarSample { perp, len });
        }
        if bars.is_empty() { return; }

        // Extend so the curve reaches the window edges.
        let first_perp = bars[0].perp - bar_width / 2.0;
        let last_perp = bars[bars.len() - 1].perp + bar_width / 2.0;

        let steps = 6usize;
        for i in 0..bars.len().saturating_sub(1) {
            let a = &bars[i];
            let b = &bars[i + 1];
            // Colors at the two endpoints.
            let ca_bot = self.color_for_perp_t(a.perp, anchor, 0.0, *gradient_rect, time, is_vertical, i, visual_num_bars, n_segments);
            let ca_top = self.color_for_perp_t(a.perp, anchor, 1.0, *gradient_rect, time, is_vertical, i, visual_num_bars, n_segments);
            let cb_bot = self.color_for_perp_t(b.perp, anchor, 0.0, *gradient_rect, time, is_vertical, i + 1, visual_num_bars, n_segments);
            let cb_top = self.color_for_perp_t(b.perp, anchor, 1.0, *gradient_rect, time, is_vertical, i + 1, visual_num_bars, n_segments);

            for s in 0..steps {
                let t1 = s as f32 / steps as f32;
                let t2 = (s + 1) as f32 / steps as f32;
                let ts1 = smoothstep(0.0, 1.0, t1);
                let ts2 = smoothstep(0.0, 1.0, t2);
                let perp1 = a.perp + (b.perp - a.perp) * t1;
                let perp2 = a.perp + (b.perp - a.perp) * t2;
                let len1 = a.len + (b.len - a.len) * ts1;
                let len2 = a.len + (b.len - a.len) * ts2;
                let c1_top = lerp_color(ca_top, cb_top, t1);
                let c2_top = lerp_color(ca_top, cb_top, t2);
                let c1_bot = lerp_color(ca_bot, cb_bot, t1);
                let c2_bot = lerp_color(ca_bot, cb_bot, t2);

                let p_top1 = bar_vertex(dir, perp1, anchor, len1, 1.0);
                let p_top2 = bar_vertex(dir, perp2, anchor, len2, 1.0);
                let p_bot1 = bar_vertex(dir, perp1, anchor, 0.0, 0.0);
                let p_bot2 = bar_vertex(dir, perp2, anchor, 0.0, 0.0);

                self.mesh.push_quad_colored(p_top1, c1_top, p_bot1, c1_bot, p_bot2, c2_bot, p_top2, c2_top);
            }
        }
        // End caps.
        {
            let a = &bars[0];
            let ca_bot = self.color_for_perp_t(a.perp, anchor, 0.0, *gradient_rect, time, is_vertical, 0, visual_num_bars, n_segments);
            let ca_top = self.color_for_perp_t(a.perp, anchor, 1.0, *gradient_rect, time, is_vertical, 0, visual_num_bars, n_segments);
            let p_top = bar_vertex(dir, a.perp, anchor, a.len, 1.0);
            let p_bot = bar_vertex(dir, a.perp, anchor, 0.0, 0.0);
            let p_edge_top = bar_vertex(dir, first_perp, anchor, a.len, 1.0);
            let p_edge_bot = bar_vertex(dir, first_perp, anchor, 0.0, 0.0);
            self.mesh.push_quad_colored(p_top, ca_top, p_bot, ca_bot, p_edge_bot, ca_bot, p_edge_top, ca_top);
        }
        {
            let a = &bars[bars.len() - 1];
            let idx = bars.len() - 1;
            let ca_bot = self.color_for_perp_t(a.perp, anchor, 0.0, *gradient_rect, time, is_vertical, idx, visual_num_bars, n_segments);
            let ca_top = self.color_for_perp_t(a.perp, anchor, 1.0, *gradient_rect, time, is_vertical, idx, visual_num_bars, n_segments);
            let p_top = bar_vertex(dir, a.perp, anchor, a.len, 1.0);
            let p_bot = bar_vertex(dir, a.perp, anchor, 0.0, 0.0);
            let p_edge_top = bar_vertex(dir, last_perp, anchor, a.len, 1.0);
            let p_edge_bot = bar_vertex(dir, last_perp, anchor, 0.0, 0.0);
            self.mesh.push_quad_colored(p_edge_top, ca_top, p_edge_bot, ca_bot, p_bot, ca_bot, p_top, ca_top);
        }
        let _ = total_size;
    }

    #[inline]
    fn color_for_perp_t(&self, perp: f32, anchor: f32, t: f32, gradient_rect: egui::Rect,
                        time: f32, is_vertical: bool, i: usize, visual_num_bars: usize,
                        _n_segments: usize) -> egui::Color32
    {
        if is_vertical {
            self.vertical_color(VisualizerDirection::Up, perp, anchor, 0.0, t, &gradient_rect, time)
        } else {
            let pos = bar_vertex(VisualizerDirection::Up, perp, anchor, 0.0, 0.0);
            self.get_color_at(i.min(visual_num_bars.saturating_sub(1)), visual_num_bars, pos.x, pos.y, t, gradient_rect, time)
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// draw_spaced helper
// ═══════════════════════════════════════════════════════════════

fn draw_spaced(ctx: &egui::Context, painter: &egui::Painter, text: &str, pos: egui::Pos2,
    font_id: &egui::FontId, align: egui::Align2, color: egui::Color32, letter_spacing: f32)
{
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
// Settings UI — same as original, no behavioural changes.
// ═══════════════════════════════════════════════════════════════

impl SettingsState {
    fn draw_settings_ui(&mut self, ui: &mut egui::Ui) {
        let scale = self.config.settings_scale.clamp(0.75, 1.5);
        egui::SidePanel::left("settings_nav")
            .resizable(false)
            .exact_width(140.0)
            .frame(egui::Frame::none().fill(egui::Color32::BLACK).inner_margin(egui::Margin::same(8.0)))
            .show_inside(ui, |ui| { self.draw_sidebar(ui); });

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
                for (_, font) in ui.style_mut().text_styles.iter_mut() { font.size *= scale; }
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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
        ui.add(egui::TextEdit::singleline(&mut self.settings_search).hint_text("Search…").desired_width(f32::INFINITY));
        ui.add_space(6.0);
        let query = self.settings_search.trim().to_lowercase();
        if !query.is_empty() {
            ui.label(egui::RichText::new("Results").small().weak());
            let mut any = false;
            for (tab, name, keywords) in SEARCH_INDEX {
                if name.to_lowercase().contains(&query) || keywords.iter().any(|k| k.contains(&query)) {
                    any = true;
                    if ui.small_button(*name).clicked() { self.settings_tab = *tab; }
                }
            }
            if !any { ui.label(egui::RichText::new("No matches").small().weak()); }
            ui.add_space(6.0); ui.separator();
        }
        ui.add_space(4.0);
        let tabs = [
            (SettingsTab::Visualizer, "Visualizer"),
            (SettingsTab::Colors, "Colors"),
            (SettingsTab::Audio, "Audio"),
            (SettingsTab::TrackInfo, "Track Info"),
            (SettingsTab::Appearance, "Appearance"),
            (SettingsTab::Performance, "Performance"),
        ];
        for (tab, label) in tabs {
            let selected = self.settings_tab == tab;
            let mut text = egui::RichText::new(label).size(13.0);
            if selected { text = text.strong().color(accent()); }
            if ui.add_sized([ui.available_width(), 26.0], egui::SelectableLabel::new(selected, text)).clicked() {
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
        }
    }

    fn tab_colors(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
        ui.horizontal_wrapped(|ui| {
            let subtabs = [
                (ColorSubTab::Single, "Single", ColorMode::StaticSingle),
                (ColorSubTab::Multi, "Multi", ColorMode::StaticMultiple),
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
        ui.add_space(4.0); ui.separator();
        match self.color_subtab {
            ColorSubTab::Single   => self.color_sub_single(ui),
            ColorSubTab::Multi    => self.color_sub_multi(ui),
            ColorSubTab::Gradient => self.color_sub_gradient(ui),
            ColorSubTab::Animated => self.color_sub_animated(ui),
        }
    }

    fn color_sub_single(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
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
        } else {
            if let Some(colors) = self.config.static_single_source.static_colors_mut() {
                if colors.is_empty() { colors.push([255, 255, 255]); }
                color_edit_reset(ui, &mut colors[0], "Bar Color", [255, 255, 255]);
            }
            ui.label("Presets:");
            let mut chosen: Option<[u8; 3]> = None;
            for chunk in self.config.color_presets.chunks(12) {
                ui.horizontal(|ui| {
                    for preset in chunk {
                        let (rect, response) = ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::click());
                        if ui.is_rect_visible(rect) {
                            let color = egui::Color32::from_rgb(preset.color[0], preset.color[1], preset.color[2]);
                            ui.painter().rect_filled(rect, 4.0, color);
                            if response.clicked() { chosen = Some(preset.color); }
                        }
                    }
                });
            }
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
        use cfg::*;
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
            });
        }
        if self.config.static_mode == StaticDistributionMode::Vertical {
            ui.horizontal(|ui| {
                ui.label("Vertical Color Style:");
                ui.radio_value(&mut self.config.vertical_color_style, VerticalColorStyle::Gradient, "Gradient");
                ui.radio_value(&mut self.config.vertical_color_style, VerticalColorStyle::Blocks, "Blocks");
            });
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
            let mut apply_colors: Option<usize> = None;
            let mut apply_full: Option<usize> = None;
            ui.horizontal_wrapped(|ui| {
                for (i, preset) in self.config.static_multi_presets.iter().enumerate() {
                    let btn = ui.button(&preset.name);
                    if btn.clicked() { apply_colors = Some(i); }
                    if btn.secondary_clicked() { apply_full = Some(i); }
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
        }
    }

    fn color_sub_gradient(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
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
        if is_dynamic {
            if let GradientSource::Dynamic { palette_size, variants_per_color, priority } = &mut self.config.gradient_source {
                Self::draw_dynamic_palette_settings(ui, "g", palette_size, variants_per_color, priority, true, true);
            }
            let preview: Vec<[u8; 3]> = self.active_palettes.gradient_stops.iter().map(|s| s.color).collect();
            Self::draw_palette_preview(ui, &preview);
        } else {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width() - 16.0, 30.0), egui::Sense::click_and_drag());
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
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut self.new_preset_name);
                if ui.button("💾 Save Preset").clicked() {
                    let name = if self.new_preset_name.trim().is_empty() { format!("Preset {}", self.config.gradient_presets.len() + 1) } else { self.new_preset_name.clone() };
                    let stops = self.config.gradient_source.static_stops().map(|s| s.to_vec()).unwrap_or_default();
                    self.config.gradient_presets.push(GradientPreset { name, stops, angle: self.config.gradient_angle, is_premade: false });
                    self.new_preset_name.clear();
                }
            });
            let mut apply_colors: Option<usize> = None;
            let mut apply_full: Option<usize> = None;
            ui.horizontal_wrapped(|ui| {
                for (i, preset) in self.config.gradient_presets.iter().enumerate() {
                    let btn = ui.button(&preset.name);
                    if btn.clicked() { apply_colors = Some(i); }
                    if btn.secondary_clicked() { apply_full = Some(i); }
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
        }
    }

    fn color_sub_animated(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
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
            let mut apply_colors: Option<usize> = None;
            let mut apply_full: Option<usize> = None;
            ui.horizontal_wrapped(|ui| {
                for (i, preset) in self.config.color_change_presets.iter().enumerate() {
                    let btn = ui.button(&preset.name);
                    if btn.clicked() { apply_colors = Some(i); }
                    if btn.secondary_clicked() { apply_full = Some(i); }
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
        }
        slider_f32_scale(ui, &mut self.config.color_change_speed, 0.0..=1.0, "Speed", 0.5);
        section_header(ui, "Motion Shape");
        combo_reset(ui, &mut self.config.color_change_motion.shape, MotionShape::default(),
                    "Shape", "motion_shape", 180.0, &MotionShape::all(), MotionShape::display_name);
        section_header(ui, "Blend");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.config.color_change_motion.blend, BlendMode::Smooth, "Smooth");
            ui.radio_value(&mut self.config.color_change_motion.blend, BlendMode::Hard, "Hard");
        });
        section_header(ui, "Motion Speed");
        slider_f32(ui, &mut self.config.color_change_motion.speed, 0.0..=2.0, "Motion Speed", 1.0);
        let shape = self.config.color_change_motion.shape;
        if matches!(shape, MotionShape::SineWave | MotionShape::TriangleWave | MotionShape::SquareWave | MotionShape::SawtoothWave | MotionShape::WaveInterference) {
            slider_f32(ui, &mut self.config.color_change_motion.wave_frequency, 0.5..=8.0, "Wave Frequency", 2.0);
        }
        if matches!(shape, MotionShape::Comet | MotionShape::BouncingBall | MotionShape::Sinelon) {
            slider_f32(ui, &mut self.config.color_change_motion.trail_length, 0.05..=0.9, "Trail Length", 0.25);
        }
        if matches!(shape, MotionShape::Stutter | MotionShape::Sparkle | MotionShape::Confetti) {
            slider_f32(ui, &mut self.config.color_change_motion.stutter_intensity, 0.0..=1.0, "Event Intensity", 0.3);
        }
        if shape.uses_intensity() {
            color_edit_reset(ui, &mut self.config.color_change_motion.off_color, "Off Color", [0, 0, 0]);
            slider_u8(ui, &mut self.config.color_change_motion.off_color_opacity, "Off-Color Opacity", 255);
        }
    }

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
        slider_f32(ui, &mut a.frequency_smoothing, 0.0..=1.0, "Frequency Smoothing", DEFAULT_FREQ_SMOOTHING);
        checkbox_reset(ui, &mut a.harmonic_emphasis, "Harmonic Emphasis", DEFAULT_HARMONIC_EMPHASIS);
        ui.horizontal(|ui| {
            ui.label("Min Frequency:");
            let r = ui.add(egui::DragValue::new(&mut a.min_frequency_hz).speed(10.0).clamp_range(10.0..=1000.0).suffix(" Hz"));
            if r.secondary_clicked() { a.min_frequency_hz = DEFAULT_MIN_FREQ; }
        });
        ui.horizontal(|ui| {
            ui.label("Max Frequency:");
            let r = ui.add(egui::DragValue::new(&mut a.max_frequency_hz).speed(100.0).clamp_range(2000.0..=24000.0).suffix(" Hz"));
            if r.secondary_clicked() { a.max_frequency_hz = DEFAULT_MAX_FREQ; }
        });
        section_header(ui, "Amplitude");
        slider_f32(ui, &mut a.sensitivity, 0.0..=100.0, "Sensitivity", DEFAULT_SENSITIVITY);
        slider_f32(ui, &mut a.raw_gain, 0.25..=4.0, "Raw Gain", DEFAULT_RAW_GAIN);
        slider_f32(ui, &mut a.response_gamma, 0.5..=2.5, "Response Curve (γ)", DEFAULT_RESPONSE_GAMMA);
        slider_f32(ui, &mut a.noise_gate, 0.0..=0.5, "Noise Gate", DEFAULT_NOISE_GATE);
        checkbox_reset(ui, &mut a.agc_enabled, "AGC / Peak Normalisation", DEFAULT_AGC_ENABLED);
        section_header(ui, "Frequency Shaping");
        slider_f32(ui, &mut a.eq_bass_db, -10.0..=10.0, "Bass EQ (< 250 Hz)", DEFAULT_EQ_BASS);
        slider_f32(ui, &mut a.eq_mid_db, -10.0..=10.0, "Mid EQ (250–4k Hz)", DEFAULT_EQ_MID);
        slider_f32(ui, &mut a.eq_treble_db, -10.0..=10.0, "Treble EQ (> 4k Hz)", DEFAULT_EQ_TREBLE);
        slider_f32(ui, &mut a.spectrum_tilt, -6.0..=6.0, "Spectrum Tilt (dB/oct)", DEFAULT_SPECTRUM_TILT);
        section_header(ui, "Timing");
        slider_f32(ui, &mut a.poll_rate_hz, 10.0..=120.0, "Poll Rate (Hz)", DEFAULT_POLL_RATE);
        section_header(ui, "Ballistics");
        if a.decay_mode != self.last_decay_mode {
            let (atk, dec) = a.decay_mode.default_ballistics();
            a.attack_time_ms = atk;
            a.decay_time_ms = dec;
            self.last_decay_mode = a.decay_mode;
        }
        combo_reset(ui, &mut a.decay_mode, DecayMode::default(), "Decay Mode", "decay_mode", 160.0, &[DecayMode::Linear, DecayMode::Exponential, DecayMode::Gravity], DecayMode::display_name);
        slider_f32(ui, &mut a.attack_time_ms, 0.0..=100.0, "Attack Time (ms)", DEFAULT_ATTACK_MS);
        slider_f32(ui, &mut a.decay_time_ms, 50.0..=2000.0, "Decay Time (ms)", DEFAULT_DECAY_MS);
        section_header(ui, "Rhythm");
        combo_reset(ui, &mut a.beat_focus, BeatFocusMode::default(), "Beat Focus", "beat_focus", 160.0, &BeatFocusMode::all(), BeatFocusMode::display_name);
        slider_f32(ui, &mut a.beat_detection_sensitivity, 0.0..=1.0, "Beat Detection Sensitivity", DEFAULT_BEAT_SENS);
        if self.last_bpm > 0.0 {
            ui.label(format!("Detected BPM: {:.1}", self.last_bpm));
        }
    }

    fn draw_audio_preset_selector(&mut self, ui: &mut egui::Ui) {
        let detected = AudioPreset::detect(&self.config.audio);
        let custom_match = self.config.custom_audio_presets.iter().find(|p| p.settings == self.config.audio);
        let current_label = if let Some(p) = detected { p.display_name().to_string() }
            else if let Some(p) = custom_match { p.name.clone() }
            else { "Custom".to_string() };
        let mut chosen_builtin: Option<AudioPreset> = None;
        ui.horizontal(|ui| {
            ui.label("Preset:");
            egui::ComboBox::from_id_source("audio_preset").selected_text(current_label).width(200.0)
                .show_ui(ui, |ui| {
                    for p in AudioPreset::all() {
                        let active = detected == Some(p);
                        if ui.selectable_label(active, p.display_name()).on_hover_text(p.description()).clicked() {
                            chosen_builtin = Some(p);
                        }
                    }
                });
        });
        if let Some(p) = chosen_builtin {
            p.apply(&mut self.config.audio);
            self.last_decay_mode = self.config.audio.decay_mode;
            self.audio_dirty = true;
        }
    }

    fn tab_track_info(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
        let system_fonts = &self.system_fonts;
        let cfg = &mut self.config.track_info;
        section_header(ui, "Cover Art");
        checkbox_reset(ui, &mut cfg.show_cover_art, "Show Cover Art", true);
        if cfg.show_cover_art {
            ui.horizontal(|ui| {
                let r = ui.add(egui::Slider::new(&mut cfg.cover_size, 40.0..=600.0).text("Size (px)"));
                if r.secondary_clicked() { cfg.cover_size = DEFAULT_COVER_SIZE; }
            });
        }
        section_header(ui, "Song Info Text");
        ui.horizontal(|ui| {
            checkbox_reset(ui, &mut cfg.show_title, "Show Title", true);
            checkbox_reset(ui, &mut cfg.show_artist, "Show Artist", true);
        });
        ui.horizontal(|ui| {
            ui.label("Font:");
            let selected_label = match &cfg.font_path {
                None => "Default (proportional)".to_string(),
                Some(p) => std::path::Path::new(p).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| p.clone()),
            };
            egui::ComboBox::from_id_source("track_font_select").selected_text(selected_label)
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
        section_header(ui, "Text Color");
        ui.horizontal(|ui| {
            ui.radio_value(&mut cfg.color_mode, TrackTextColorMode::Static, "Static");
            ui.radio_value(&mut cfg.color_mode, TrackTextColorMode::Visualizer, "Visualizer Color");
        });
        if cfg.color_mode == TrackTextColorMode::Static {
            color_edit_reset(ui, &mut cfg.static_color, "Color", [220, 220, 220]);
        }
        slider_u8(ui, &mut cfg.opacity, "Text Opacity", 255);
    }

    fn tab_appearance(&mut self, ui: &mut egui::Ui) {
        use cfg::*;
        let show_help = self.config.show_help_text;
        section_header(ui, "Background");
        color_edit_reset(ui, &mut self.config.bg_color, "Background Color", [0, 0, 0]);
        slider_u8(ui, &mut self.config.bg_opacity, "Background Opacity", DEFAULT_BG_OPACITY);
        section_header(ui, "Menu");
        ui.horizontal(|ui| {
            ui.label("Menu Scale:");
            let mut current_scale = self.config.settings_scale;
            egui::ComboBox::from_id_source("menu_scale").selected_text(format!("{:.0}%", current_scale * 100.0))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut current_scale, 0.75, "75%");
                    ui.selectable_value(&mut current_scale, 1.00, "100%");
                    ui.selectable_value(&mut current_scale, 1.25, "125%");
                    ui.selectable_value(&mut current_scale, 1.50, "150%");
                });
            self.config.settings_scale = current_scale;
        });
        checkbox_reset(ui, &mut self.config.open_settings_on_launch, "Open Settings on Launch", DEFAULT_OPEN_SETTINGS_ON_LAUNCH);
        checkbox_reset(ui, &mut self.config.show_help_text, "Show Inline Help Text", DEFAULT_SHOW_HELP_TEXT);
        let _ = show_help;
    }

    fn tab_performance(&mut self, ui: &mut egui::Ui) {
        section_header(ui, "Display");
        checkbox_reset(ui, &mut self.config.show_fps, "Show FPS Counter", false);
        ui.label(format!("Current FPS: {:.1}", self.current_fps));
        ui.add_space(6.0);
    }

    fn draw_dynamic_palette_settings(ui: &mut egui::Ui, id_suffix: &str, palette_size: &mut u8, variants: &mut u8, priority: &mut PalettePriority, show_size: bool, show_variants: bool) {
        ui.push_id(id_suffix, |ui| {
            ui.horizontal(|ui| {
                ui.label("Priority:");
                egui::ComboBox::from_id_source("priority").selected_text(priority.display_name())
                    .show_ui(ui, |ui| {
                        for p in PalettePriority::all() { ui.selectable_value(priority, p, p.display_name()); }
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
        egui::ComboBox::from_id_source("fft_size").selected_text(format!("{}", a.fft_size))
            .show_ui(ui, |ui| {
                for size in [512usize, 1024, 2048, 4096, 8192] {
                    ui.selectable_value(&mut a.fft_size, size, format!("{size}"));
                }
            });
    });
}

// ═══════════════════════════════════════════════════════════════
// Settings viewport (deferred)
// ═══════════════════════════════════════════════════════════════

fn run_settings_viewport(
    ctx: &egui::Context,
    shared: &Mutex<SettingsState>,
    open: &AtomicBool,
    generation: &AtomicU64,
) {
    let mut st = lock_state(shared);
    if ctx.input(|i| i.viewport().close_requested()) {
        st.config.save();
        drop(st);
        open.store(false, Ordering::Relaxed);
        ctx.request_repaint_of(egui::ViewportId::ROOT);
        return;
    }
    if (st.config.settings_scale - st.old_settings_scale).abs() > 0.01 {
        st.old_settings_scale = st.config.settings_scale;
        let scale = st.config.settings_scale.clamp(0.75, 1.5);
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(560.0 * scale, 720.0 * scale)));
    }
    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(egui::Color32::BLACK))
        .show(ctx, |ui| { st.draw_settings_ui(ui); });

    if let Some(idx) = st.pending_audio_preset_delete {
        egui::Window::new("Confirm Delete").collapsible(false).resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                let name = st.config.custom_audio_presets.get(idx).map(|p| p.name.clone()).unwrap_or_default();
                ui.label(format!("Delete preset '{}'?", name));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Yes").clicked() {
                        if idx < st.config.custom_audio_presets.len() {
                            st.config.custom_audio_presets.remove(idx);
                        }
                        st.pending_audio_preset_delete = None;
                    }
                    if ui.button("No").clicked() { st.pending_audio_preset_delete = None; }
                });
            });
    }

    let fp = config_fingerprint(&st.config);
    let changed = st.audio_dirty || fp != st.last_fingerprint;
    st.last_fingerprint = fp;
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

// ═══════════════════════════════════════════════════════════════
// eframe::App
// ═══════════════════════════════════════════════════════════════

impl eframe::App for VisualizerApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] { egui::Rgba::TRANSPARENT.to_array() }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        if dt > 0.0 && dt < 0.1 {
            let instant_fps = 1.0 / dt;
            self.current_fps = if self.current_fps <= 0.0 { instant_fps } else { self.current_fps * 0.9 + instant_fps * 0.1 };
        }
        if ctx.pixels_per_point() != 1.0 { ctx.set_pixels_per_point(1.0); }

        // ── Commands from tray / hotkey threads ─────────────
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

        // ── Pick up config edits ────────────────────────────
        let generation = self.config_gen.load(Ordering::Acquire);
        if generation != self.seen_gen {
            self.seen_gen = generation;
            let mut st = lock_state(&self.shared);
            self.config = st.config.clone();
            if st.audio_dirty || self.config.audio != *self.audio_settings.load_full() {
                self.audio_settings.store(Arc::new(self.config.audio.clone()));
                st.audio_dirty = false;
            }
        }

        self.poll_track_updates(ctx);

        let new_hash = self.palette_fingerprint();
        if new_hash != self.last_palette_hash {
            self.update_palettes();
            self.last_palette_hash = new_hash;
            let mut st = lock_state(&self.shared);
            st.active_palettes = self.active_palettes.clone();
            drop(st);
            if self.settings_open.load(Ordering::Relaxed) {
                ctx.request_repaint_of(egui::ViewportId::from_hash_of("settings"));
            }
        }

        if self.config.track_info.font_path != self.loaded_font_path {
            self.reload_custom_font(ctx);
        }

        let n_bars = self.config.audio.num_bars;
        if self.current_bars.len() != n_bars {
            self.current_bars.resize(n_bars, 0.0);
            self.current_peaks.resize(n_bars, 0.0);
        }

        let mut latest: Option<BarFrame> = None;
        while let Ok(frame) = self.audio_rx.try_recv() { latest = Some(frame); }
        if let Some(frame) = latest {
            let limit = self.current_bars.len().min(frame.values.len());
            self.current_bars[..limit].copy_from_slice(&frame.values[..limit]);
            self.current_peaks[..limit].copy_from_slice(&frame.peaks[..limit]);
            self.last_bpm = frame.bpm;
        }

        // ── Mouse passthrough (set once) ─────────────────────
        if !self.is_passthrough_set {
            ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(true));
            self.is_passthrough_set = true;
        }

        // ── Capture monitor rect once ────────────────────────
        if self.monitor_rect.is_none() {
            let sr = ctx.screen_rect();
            if sr.width() > 100.0 && sr.height() > 100.0 {
                self.monitor_rect = Some(sr);
            }
        }
        let monitor = self.monitor_rect.unwrap_or_else(|| ctx.screen_rect());

        // ── Compute content bbox & shrink window ─────────────
        let target = self.compute_overlay_rect(monitor);

        // Resize only when the required size changes meaningfully to
        // avoid a resize storm. Position is applied whenever the top-left
        // moves.
        let cur_size = self.overlay_rect.size();
        let want_size = target.size();
        let size_changed = (cur_size - want_size).length() > 32.0;
        let pos_changed = (self.overlay_rect.min - target.min).length() > 1.0;

        if size_changed {
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(want_size));
        }
        if size_changed || pos_changed {
            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(target.min));
            self.overlay_rect = target;
        }

        // Draw into the *desired* local rect (0..want_size). For the one
        // frame where the OS hasn't resized yet, egui clips naturally.
        let local_rect = egui::Rect::from_min_size(egui::Pos2::ZERO, want_size);
        let overlay_origin = target.min;
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("overlay")));

        if self.config.bg_opacity > 0 {
            let c = self.config.bg_color;
            let bg_color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], self.config.bg_opacity);
            painter.rect_filled(local_rect, 0.0, bg_color);
        }

        self.draw_bars(&painter, ctx, local_rect, monitor.height(), monitor.width());
        self.draw_track_info(&painter, ctx, local_rect, overlay_origin, monitor);

        if self.config.show_fps {
            let fps_rect_pos = egui::pos2(20.0, 20.0);
            // Only draw if the fps position is inside our local window.
            let world_pos = monitor.left_top() + fps_rect_pos.to_vec2();
            if local_rect.contains(world_pos - overlay_origin.to_vec2()) {
                painter.text(
                    world_pos - overlay_origin.to_vec2(),
                    egui::Align2::LEFT_TOP,
                    format!("FPS: {:.1}", self.current_fps),
                    egui::FontId::monospace(16.0),
                    egui::Color32::GREEN,
                );
            }
        }

        // ── Settings viewport ───────────────────────────────
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
            if self.settings_icon.width > 0 { builder = builder.with_icon(self.settings_icon.clone()); }

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

        // ── Repaint policy ──────────────────────────────────
        // Only request frames while something is visibly moving.
        let moving = self.current_bars.iter().any(|&v| v > 0.0005)
            || self.current_peaks.iter().any(|&v| v > 0.0005)
            || self.config.color_mode == ColorMode::Animated;
        if moving { ctx.request_repaint(); }
    }

    fn on_exit(&mut self, _gl: Option<&glow::Context>) { self.config.save(); }
}

// ═══════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════

fn main() -> eframe::Result<()> {
    let (egui_icon, _) = load_icon();

    // Give the window a sensible initial size — the app shrinks/grows it
    // from here to match the content.
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_transparent(true)
            .with_decorations(false)
            .with_always_on_top()
            .with_inner_size([1280.0, 720.0])
            .with_position([0.0, 0.0])
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
