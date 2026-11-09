use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────
// Default value constants (single source of truth)
// ─────────────────────────────────────────────────────────────

pub const DEFAULT_BAR_WIDTH:       f32 = 11.0;
pub const DEFAULT_BAR_GAP:         f32 = 1.0;
pub const DEFAULT_BAR_RADIUS:      f32 = 2.0;
pub const DEFAULT_HEIGHT_SCALE:    f32 = 0.65;
pub const DEFAULT_NUM_BARS:        usize = 80;
pub const DEFAULT_MAX_BAR_HEIGHT:  f32 = 0.0;
pub const DEFAULT_X_OFFSET:        f32 = 0.0;
pub const DEFAULT_Y_OFFSET:        f32 = 0.0;
pub const DEFAULT_BG_OPACITY:      u8 = 0;
pub const DEFAULT_VISUALIZER_OPACITY: u8 = 255;
pub const DEFAULT_COVER_OPACITY:   u8 = 255;
pub const DEFAULT_SETTINGS_SCALE:  f32 = 1.0;
pub const DEFAULT_OPEN_SETTINGS_ON_LAUNCH: bool = true;
pub const DEFAULT_SHOW_HELP_TEXT:  bool = true;
pub const DEFAULT_TITLE_FONT_SIZE:  f32 = 24.0;
pub const DEFAULT_ARTIST_FONT_SIZE: f32 = 30.0;
pub const DEFAULT_COVER_SIZE:       f32 = 190.0;
pub const DEFAULT_FFT_SIZE:                usize = 2048;
pub const DEFAULT_MIN_FREQ:                f32   = 30.0;
pub const DEFAULT_MAX_FREQ:                f32   = 16_000.0;
pub const DEFAULT_MIN_DB:                  f32   = -85.0;
pub const DEFAULT_MAX_DB:                  f32   = -20.0;
pub const DEFAULT_TARGET_RANGE:            f32   = 60.0;
pub const DEFAULT_SENSITIVITY:             f32   = 50.0;
pub const DEFAULT_RAW_GAIN:                f32   = 1.0;
pub const DEFAULT_RESPONSE_GAMMA:          f32   = 1.5;
pub const DEFAULT_SPECTRUM_TILT:           f32   = 2.0;
pub const DEFAULT_EQ_BASS:                 f32   = 0.0;
pub const DEFAULT_EQ_MID:                  f32   = 0.0;
pub const DEFAULT_EQ_TREBLE:               f32   = 0.0;
pub const DEFAULT_NOISE_GATE:              f32   = 0.0;
pub const DEFAULT_AGC_ENABLED:             bool  = true;
pub const DEFAULT_AGC_ATTACK:              f32   = 0.15;
pub const DEFAULT_AGC_RELEASE:             f32   = 0.005;
pub const DEFAULT_POLL_RATE:               f32   = 60.0;
pub const DEFAULT_ATTACK_MS:               f32   = 5.0;
pub const DEFAULT_DECAY_MS:                f32   = 250.0;
pub const DEFAULT_AVERAGE_SIZE:            usize = 1;
pub const DEFAULT_PEAK_HOLD_MS:            f32   = 0.0;
pub const DEFAULT_GRAVITY_COEFF:           f32   = 1.0;
pub const DEFAULT_BEAT_SENS:               f32   = 0.5;
pub const DEFAULT_ENERGY_THRESHOLD:        f32   = 1.5;
pub const DEFAULT_BPM_PHASE_SMOOTHING:     f32   = 0.3;
pub const DEFAULT_FREQ_SMOOTHING:          f32   = 0.0;
pub const DEFAULT_DC_BLOCK:                bool  = true;
pub const DEFAULT_PER_BAND_BALLISTICS:     bool  = false;
pub const DEFAULT_PER_FRAME_NORMALIZATION: bool  = false;
pub const DEFAULT_BEAT_ENVELOPE_DECAY_MS:  f32   = 120.0;
pub const DEFAULT_HARMONIC_EMPHASIS:       bool  = false;

// ─────────────────────────────────────────────────────────────
// Visualizer primitives
// ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum VisualizerDirection { Up, Down, Left, Right }
impl Default for VisualizerDirection { fn default() -> Self { VisualizerDirection::Up } }

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrequencyDirection { LeftToRight, RightToLeft, Mirrored }
impl Default for FrequencyDirection { fn default() -> Self { FrequencyDirection::LeftToRight } }

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode {
    #[serde(alias = "Dynamic")] StaticSingle,
    StaticMultiple,
    Gradient,
    #[serde(alias = "ColorChanging")] Animated,
}
impl Default for ColorMode { fn default() -> Self { ColorMode::StaticSingle } }

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum StaticDistributionMode { Blocks, Sequential, Random, Vertical }
impl Default for StaticDistributionMode { fn default() -> Self { StaticDistributionMode::Blocks } }

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum VerticalColorStyle { Gradient, Blocks }
impl Default for VerticalColorStyle { fn default() -> Self { VerticalColorStyle::Gradient } }
impl VerticalColorStyle {
    pub fn display_name(self) -> &'static str {
        match self {
            VerticalColorStyle::Gradient => "Gradient (Smooth Blend)",
            VerticalColorStyle::Blocks => "Blocks (Distinct Sections)",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            VerticalColorStyle::Gradient => "Smoothly blends all colors vertically across the bar.",
            VerticalColorStyle::Blocks => "Divides the bar into solid, distinct color sections (classic style).",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GradientStop { pub color: [u8; 3], pub position: f32 }

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ColorPreset {
    pub color: [u8; 3],
    #[serde(default)] pub is_premade: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StaticMultiPreset {
    pub name: String, pub colors: Vec<[u8; 3]>,
    pub mode: StaticDistributionMode,
    #[serde(default = "default_sequential_bars_per_color")] pub bars_per_color: usize,
    #[serde(default)] pub is_premade: bool,
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GradientPreset {
    pub name: String, pub stops: Vec<GradientStop>, pub angle: f32,
    #[serde(default)] pub is_premade: bool,
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ColorChangePreset {
    pub name: String, pub colors: Vec<[u8; 3]>, pub speed: f32,
    #[serde(default)] pub is_premade: bool,
    #[serde(default)] pub motion: MotionSettings,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PalettePriority { MostCommon, MostCommonLight, Vivid }
impl Default for PalettePriority { fn default() -> Self { PalettePriority::MostCommon } }
impl PalettePriority {
    pub fn display_name(self) -> &'static str {
        match self {
            PalettePriority::MostCommon => "Most Common",
            PalettePriority::MostCommonLight => "Most Common Light",
            PalettePriority::Vivid => "Vivid / High Contrast",
        }
    }
    pub fn all() -> [PalettePriority; 3] {
        [PalettePriority::MostCommon, PalettePriority::MostCommonLight, PalettePriority::Vivid]
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum ColorSource {
    Static(Vec<[u8; 3]>),
    Dynamic {
        #[serde(default = "default_palette_size")] palette_size: u8,
        #[serde(default)] variants_per_color: u8,
        #[serde(default)] priority: PalettePriority,
    },
}
impl Default for ColorSource { fn default() -> Self { ColorSource::Static(default_static_colors()) } }
impl ColorSource {
    pub fn is_dynamic(&self) -> bool { matches!(self, ColorSource::Dynamic { .. }) }
    pub fn static_colors(&self) -> Option<&[[u8; 3]]> {
        match self { ColorSource::Static(v) => Some(v.as_slice()), _ => None }
    }
    pub fn static_colors_mut(&mut self) -> Option<&mut Vec<[u8; 3]>> {
        match self { ColorSource::Static(v) => Some(v), _ => None }
    }
    pub fn dynamic_settings(&self) -> Option<(u8, u8, PalettePriority)> {
        match self {
            ColorSource::Dynamic { palette_size, variants_per_color, priority } =>
                Some((*palette_size, *variants_per_color, *priority)),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum GradientSource {
    Static(Vec<GradientStop>),
    Dynamic {
        #[serde(default = "default_palette_size")] palette_size: u8,
        #[serde(default)] variants_per_color: u8,
        #[serde(default)] priority: PalettePriority,
    },
}
impl Default for GradientSource { fn default() -> Self { GradientSource::Static(default_gradient_stops()) } }
impl GradientSource {
    pub fn is_dynamic(&self) -> bool { matches!(self, GradientSource::Dynamic { .. }) }
    pub fn static_stops(&self) -> Option<&[GradientStop]> {
        match self { GradientSource::Static(s) => Some(s.as_slice()), _ => None }
    }
    pub fn static_stops_mut(&mut self) -> Option<&mut Vec<GradientStop>> {
        match self { GradientSource::Static(s) => Some(s), _ => None }
    }
    pub fn dynamic_settings(&self) -> Option<(u8, u8, PalettePriority)> {
        match self {
            GradientSource::Dynamic { palette_size, variants_per_color, priority } =>
                Some((*palette_size, *variants_per_color, *priority)),
            _ => None,
        }
    }
}
fn default_palette_size() -> u8 { 1 }

// ─────────────────────────────────────────────────────────────
// Motion
// ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum MotionShape {
    Uniform, HorizontalWipe, VerticalWipe, DiagonalWipe,
    RadialWipe, Spiral, WaveInterference, Ripple,
    SineWave, TriangleWave, SquareWave, SawtoothWave,
    Strobe, Pulse, Heartbeat, Stutter, Comet, Chase, BouncingBall, Sparkle,
    FireSimulation, Plasma, RainbowCycle, Confetti, Sinelon,
    Breathing, FlipFlop, Kaleidoscope, DigitalRain, MeteorShower,
}
impl Default for MotionShape { fn default() -> Self { MotionShape::Uniform } }
impl MotionShape {
    pub fn display_name(self) -> &'static str {
        match self {
            MotionShape::Uniform => "Uniform",
            MotionShape::HorizontalWipe => "Horizontal Wipe",
            MotionShape::VerticalWipe => "Vertical Wipe",
            MotionShape::DiagonalWipe => "Diagonal Wipe",
            MotionShape::RadialWipe => "Radial Wipe",
            MotionShape::Spiral => "Spiral",
            MotionShape::WaveInterference => "Wave Interference",
            MotionShape::Ripple => "Ripple",
            MotionShape::SineWave => "Sine Wave",
            MotionShape::TriangleWave => "Triangle Wave",
            MotionShape::SquareWave => "Square Wave",
            MotionShape::SawtoothWave => "Sawtooth Wave",
            MotionShape::Strobe => "Strobe",
            MotionShape::Pulse => "Pulse",
            MotionShape::Heartbeat => "Heartbeat",
            MotionShape::Stutter => "Stutter",
            MotionShape::Comet => "Comet",
            MotionShape::Chase => "Chase",
            MotionShape::BouncingBall => "Bouncing Ball",
            MotionShape::Sparkle => "Sparkle",
            MotionShape::FireSimulation => "Fire Simulation",
            MotionShape::Plasma => "Plasma",
            MotionShape::RainbowCycle => "Rainbow Cycle",
            MotionShape::Confetti => "Confetti",
            MotionShape::Sinelon => "Sinelon",
            MotionShape::Breathing => "Breathing",
            MotionShape::FlipFlop => "Flip Flop",
            MotionShape::Kaleidoscope => "Kaleidoscope",
            MotionShape::DigitalRain => "Digital Rain",
            MotionShape::MeteorShower => "Meteor Shower",
        }
    }
    pub fn uses_vertical_axis(self) -> bool {
        matches!(self,
            MotionShape::VerticalWipe | MotionShape::DiagonalWipe
            | MotionShape::RadialWipe | MotionShape::Spiral
            | MotionShape::WaveInterference | MotionShape::Ripple
            | MotionShape::Plasma | MotionShape::Kaleidoscope
            | MotionShape::DigitalRain | MotionShape::MeteorShower
        )
    }
    pub fn uses_intensity(self) -> bool {
        matches!(
            self,
            MotionShape::Strobe | MotionShape::Pulse | MotionShape::Heartbeat
            | MotionShape::Comet | MotionShape::Chase | MotionShape::BouncingBall
            | MotionShape::Sparkle | MotionShape::Sinelon
            | MotionShape::FireSimulation | MotionShape::Confetti
            | MotionShape::Breathing | MotionShape::FlipFlop | MotionShape::DigitalRain | MotionShape::MeteorShower
        )
    }
    pub fn all() -> [MotionShape; 30] {
        [
            MotionShape::Uniform, MotionShape::HorizontalWipe, MotionShape::VerticalWipe,
            MotionShape::DiagonalWipe, MotionShape::RadialWipe, MotionShape::Spiral,
            MotionShape::WaveInterference, MotionShape::Ripple,
            MotionShape::SineWave, MotionShape::TriangleWave, MotionShape::SquareWave,
            MotionShape::SawtoothWave, MotionShape::Strobe, MotionShape::Pulse,
            MotionShape::Heartbeat, MotionShape::Stutter, MotionShape::Comet,
            MotionShape::Chase, MotionShape::BouncingBall, MotionShape::Sparkle,
            MotionShape::FireSimulation, MotionShape::Plasma, MotionShape::RainbowCycle,
            MotionShape::Confetti, MotionShape::Sinelon, MotionShape::Breathing,
            MotionShape::FlipFlop, MotionShape::Kaleidoscope, MotionShape::DigitalRain,
            MotionShape::MeteorShower,
        ]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendMode { Smooth, Hard }
impl Default for BlendMode { fn default() -> Self { BlendMode::Smooth } }

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MotionSettings {
    pub shape: MotionShape, pub blend: BlendMode, pub speed: f32,
    pub trail_length: f32, pub stutter_intensity: f32,
    pub wave_frequency: f32, pub wipe_softness: f32,
    #[serde(default)] pub off_color: [u8; 3],
    #[serde(default = "default_off_color_opacity")] pub off_color_opacity: u8,
}
fn default_off_color_opacity() -> u8 { 255 }
impl Default for MotionSettings {
    fn default() -> Self {
        Self {
            shape: MotionShape::Uniform, blend: BlendMode::Smooth, speed: 1.0,
            trail_length: 0.25, stutter_intensity: 0.3,
            wave_frequency: 2.0, wipe_softness: 0.1,
            off_color: [0, 0, 0], off_color_opacity: 255,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Track info
// ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrackTextColorMode { Static, Visualizer }
impl Default for TrackTextColorMode { fn default() -> Self { TrackTextColorMode::Static } }

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrackTextAlign { Left, Center, Right }
impl Default for TrackTextAlign { fn default() -> Self { TrackTextAlign::Right } }

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TrackInfoConfig {
    pub show_cover_art: bool, pub cover_size: f32,
    pub cover_x_offset: f32, pub cover_y_offset: f32,
    #[serde(default = "default_cover_opacity")] pub cover_opacity: u8,
    pub show_title: bool, pub show_artist: bool,
    pub title_font_size: f32, pub artist_font_size: f32,
    #[serde(default, alias = "custom_font_path")] pub font_path: Option<String>,
    pub color_mode: TrackTextColorMode, pub static_color: [u8; 3], pub opacity: u8,
    pub letter_spacing: f32, pub line_height: f32, pub alignment: TrackTextAlign,
    pub text_x_offset: f32, pub text_y_offset: f32,
    pub shadow_enabled: bool, pub shadow_offset_x: f32, pub shadow_offset_y: f32,
    pub shadow_blur: f32, pub shadow_color: [u8; 3],
    pub outline_enabled: bool, pub outline_thickness: f32, pub outline_color: [u8; 3],
}
fn default_cover_opacity() -> u8 { DEFAULT_COVER_OPACITY }
impl Default for TrackInfoConfig {
    fn default() -> Self {
        Self {
            show_cover_art: true,
            cover_size: DEFAULT_COVER_SIZE,
            cover_x_offset: 0.0, cover_y_offset: 0.0,
            cover_opacity: DEFAULT_COVER_OPACITY,
            show_title: true, show_artist: true,
            title_font_size: DEFAULT_TITLE_FONT_SIZE,
            artist_font_size: DEFAULT_ARTIST_FONT_SIZE,
            font_path: None, color_mode: TrackTextColorMode::Static,
            static_color: [220, 220, 220], opacity: 255,
            letter_spacing: 0.0, line_height: 1.3, alignment: TrackTextAlign::Right,
            text_x_offset: 0.0, text_y_offset: 0.0,
            shadow_enabled: true, shadow_offset_x: 2.0, shadow_offset_y: 2.0,
            shadow_blur: 0.0, shadow_color: [0, 0, 0],
            outline_enabled: false, outline_thickness: 1.0, outline_color: [0, 0, 0],
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Audio: window, scaling, decay, channel, beat
// ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowFunction { Hann, Hamming, BlackmanHarris, FlatTop, Rectangular }
impl Default for WindowFunction { fn default() -> Self { WindowFunction::Hann } }
impl WindowFunction {
    pub fn display_name(self) -> &'static str {
        match self {
            WindowFunction::Hann => "Hann",
            WindowFunction::Hamming => "Hamming",
            WindowFunction::BlackmanHarris => "Blackman-Harris",
            WindowFunction::FlatTop => "Flat Top",
            WindowFunction::Rectangular => "Rectangular",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            WindowFunction::Hann => "Balanced default. Good amplitude accuracy with low spectral leakage.",
            WindowFunction::Hamming => "Slightly narrower main lobe than Hann, less leakage suppression.",
            WindowFunction::BlackmanHarris => "Strongest side-lobe suppression. Tones look cleanest, at the cost of a wider main lobe.",
            WindowFunction::FlatTop => "Best amplitude accuracy. Peaks read true, but neighbouring bins blur more.",
            WindowFunction::Rectangular => "No tapering. Sharpest time resolution, but leaks badly on non-integer-period tones.",
        }
    }
    pub fn all() -> [WindowFunction; 5] {
        [WindowFunction::Hann, WindowFunction::Hamming, WindowFunction::BlackmanHarris, WindowFunction::FlatTop, WindowFunction::Rectangular]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrequencyScaling { Linear, Logarithmic }
impl Default for FrequencyScaling { fn default() -> Self { FrequencyScaling::Logarithmic } }
impl FrequencyScaling {
    pub fn display_name(self) -> &'static str {
        match self {
            FrequencyScaling::Linear => "Linear",
            FrequencyScaling::Logarithmic => "Logarithmic",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            FrequencyScaling::Linear => "Even Hz spacing. Favours treble; bass compresses into a few bars.",
            FrequencyScaling::Logarithmic => "Matches human hearing. Gives bass and mids more visual space.",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecayMode { Linear, Exponential, Gravity }
impl Default for DecayMode { fn default() -> Self { DecayMode::Exponential } }
impl DecayMode {
    pub fn display_name(self) -> &'static str {
        match self {
            DecayMode::Linear => "Linear",
            DecayMode::Exponential => "Exponential",
            DecayMode::Gravity => "Gravity",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            DecayMode::Linear => "Steady fall at a constant rate. Predictable, slightly mechanical.",
            DecayMode::Exponential => "Fast initial drop that slows near the bottom. Natural-looking.",
            DecayMode::Gravity => "Accelerates downward like a falling object. Punchy, physical.",
        }
    }
    pub fn default_ballistics(self) -> (f32, f32) {
        match self {
            DecayMode::Linear => (20.0, 400.0),
            DecayMode::Exponential => (5.0, 300.0),
            DecayMode::Gravity => (0.0, 500.0),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChannelMixMode { Mono, Left, Right, Mid, Side }
impl Default for ChannelMixMode { fn default() -> Self { ChannelMixMode::Mono } }
impl ChannelMixMode {
    pub fn display_name(self) -> &'static str {
        match self {
            ChannelMixMode::Mono => "Mono (L+R)",
            ChannelMixMode::Left => "Left Channel",
            ChannelMixMode::Right => "Right Channel",
            ChannelMixMode::Mid => "Mid (Centered)",
            ChannelMixMode::Side => "Side (Stereo Width)",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            ChannelMixMode::Mono => "Standard L+R mix. Best for balanced visualizers.",
            ChannelMixMode::Left => "Only uses the left channel. Useful for checking stereo panning.",
            ChannelMixMode::Right => "Only uses the right channel.",
            ChannelMixMode::Mid => "Extracts the center (L+R). Highlights vocals and kick drums.",
            ChannelMixMode::Side => "Extracts the stereo width (L-R). Highlights reverbs and wide synths.",
        }
    }
    pub fn all() -> [ChannelMixMode; 5] {
        [ChannelMixMode::Mono, ChannelMixMode::Left, ChannelMixMode::Right, ChannelMixMode::Mid, ChannelMixMode::Side]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum BeatFocusMode { Bass, Mid, Treble }
impl Default for BeatFocusMode { fn default() -> Self { BeatFocusMode::Bass } }
impl BeatFocusMode {
    pub fn display_name(self) -> &'static str {
        match self {
            BeatFocusMode::Bass => "Bass (Kick)",
            BeatFocusMode::Mid => "Mid (Snare)",
            BeatFocusMode::Treble => "Treble (Hi-hat)",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            BeatFocusMode::Bass => "Detects kicks and sub-bass (20-250 Hz). Best for EDM, Hip-hop, Pop.",
            BeatFocusMode::Mid => "Detects snares and vocals (250-4000 Hz). Best for Rock, Jazz.",
            BeatFocusMode::Treble => "Detects hi-hats and cymbals (4000+ Hz). Best for fast, complex genres.",
        }
    }
    pub fn all() -> [BeatFocusMode; 3] {
        [BeatFocusMode::Bass, BeatFocusMode::Mid, BeatFocusMode::Treble]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum BarInterpolation { Step, Smooth }
impl Default for BarInterpolation { fn default() -> Self { BarInterpolation::Step } }
impl BarInterpolation {
    pub fn display_name(self) -> &'static str {
        match self {
            BarInterpolation::Step => "Step (Blocks)",
            BarInterpolation::Smooth => "Smooth (Fluid Wave)",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            BarInterpolation::Step => "Classic rigid rectangles. Precise frequency reading.",
            BarInterpolation::Smooth => "Draws a filled curve between the bars. Looks like a fluid wave.",
        }
    }
    pub fn all() -> [BarInterpolation; 2] {
        [BarInterpolation::Step, BarInterpolation::Smooth]
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightingCurve { None, AWeighting, KWeighting }
impl Default for WeightingCurve { fn default() -> Self { WeightingCurve::None } }
impl WeightingCurve {
    pub fn display_name(self) -> &'static str {
        match self {
            WeightingCurve::None => "Flat (None)",
            WeightingCurve::AWeighting => "A-Weighting (Perceptual)",
            WeightingCurve::KWeighting => "K-Weighting (Broadcast)",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            WeightingCurve::None => "No weighting. Every frequency is treated equally.",
            WeightingCurve::AWeighting => "Standard A-weighting curve. Rolls off deep bass and boosts presence. Sounds more natural.",
            WeightingCurve::KWeighting => "ITU-R BS.1770 K-weighting. Broadcast-style high-shelf and high-pass. Good for speech.",
        }
    }
    pub fn all() -> [WeightingCurve; 3] {
        [WeightingCurve::None, WeightingCurve::AWeighting, WeightingCurve::KWeighting]
    }
}

// ─────────────────────────────────────────────────────────────
// Audio settings
// ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AudioSettings {
    // Signal processing
    #[serde(default = "default_fft_size")]             pub fft_size: usize,
    #[serde(default)]                                  pub window_function: WindowFunction,
    #[serde(default = "default_num_bars")]             pub num_bars: usize,
    #[serde(default)]                                  pub frequency_scaling: FrequencyScaling,
    #[serde(default = "default_min_freq")]             pub min_frequency_hz: f32,
    #[serde(default = "default_max_freq")]             pub max_frequency_hz: f32,
    #[serde(default)]                                  pub channel_mix: ChannelMixMode,
    #[serde(default)]                                  pub weighting: WeightingCurve,
    #[serde(default = "default_dc_block")]             pub dc_block: bool,
    #[serde(default = "default_freq_smoothing")]       pub frequency_smoothing: f32,
    #[serde(default = "default_harmonic_emphasis")]    pub harmonic_emphasis: bool,

    // Amplitude / dynamics
    #[serde(default = "default_sensitivity")]          pub sensitivity: f32,
    #[serde(default = "default_raw_gain")]             pub raw_gain: f32,
    #[serde(default = "default_response_gamma")]       pub response_gamma: f32,
    #[serde(default = "default_noise_gate")]           pub noise_gate: f32,
    #[serde(default = "default_min_db")]               pub min_decibels: f32,
    #[serde(default = "default_max_db")]               pub max_decibels: f32,
    #[serde(default = "default_target_range")]         pub target_dynamic_range_db: f32,
    #[serde(default = "default_per_frame_normalization")] pub per_frame_normalization: bool,
    #[serde(default = "default_agc_enabled")]          pub agc_enabled: bool,
    #[serde(default = "default_agc_attack")]           pub agc_attack: f32,
    #[serde(default = "default_agc_release")]          pub agc_release: f32,

    // Frequency shaping
    #[serde(default = "default_spectrum_tilt")]        pub spectrum_tilt: f32,
    #[serde(default = "default_eq_bass")]              pub eq_bass_db: f32,
    #[serde(default = "default_eq_mid")]               pub eq_mid_db: f32,
    #[serde(default = "default_eq_treble")]            pub eq_treble_db: f32,

    // Timing
    #[serde(default = "default_poll_rate_hz")]         pub poll_rate_hz: f32,
    #[serde(default = "default_attack_ms")]            pub attack_time_ms: f32,
    #[serde(default = "default_decay_ms")]             pub decay_time_ms: f32,
    #[serde(default)]                                  pub decay_mode: DecayMode,
    #[serde(default = "default_average_size")]         pub average_size: usize,
    #[serde(default = "default_peak_hold_ms")]         pub peak_hold_ms: f32,
    #[serde(default = "default_gravity_coeff")]        pub gravity_coefficient: f32,

    // Per-band ballistics
    #[serde(default = "default_per_band_ballistics")]  pub per_band_ballistics: bool,
    #[serde(default = "default_attack_ms")]            pub bass_attack_ms: f32,
    #[serde(default = "default_decay_ms")]             pub bass_decay_ms: f32,
    #[serde(default = "default_attack_ms")]            pub mid_attack_ms: f32,
    #[serde(default = "default_decay_ms")]             pub mid_decay_ms: f32,
    #[serde(default = "default_attack_ms")]            pub treble_attack_ms: f32,
    #[serde(default = "default_decay_ms")]             pub treble_decay_ms: f32,

    // Rhythm
    #[serde(default)]                                  pub beat_focus: BeatFocusMode,
    #[serde(default = "default_beat_sensitivity")]     pub beat_detection_sensitivity: f32,
    #[serde(default = "default_energy_threshold")]     pub energy_threshold_multiplier: f32,
    #[serde(default)]                                  pub bpm_lock_enabled: bool,
    #[serde(default = "default_bpm_phase_smoothing")]  pub bpm_phase_smoothing: f32,
    #[serde(default = "default_beat_envelope_decay")]  pub beat_envelope_decay_ms: f32,
}

fn default_fft_size() -> usize { DEFAULT_FFT_SIZE }
fn default_num_bars() -> usize { DEFAULT_NUM_BARS }
fn default_min_freq() -> f32 { DEFAULT_MIN_FREQ }
fn default_max_freq() -> f32 { DEFAULT_MAX_FREQ }
fn default_min_db() -> f32 { DEFAULT_MIN_DB }
fn default_max_db() -> f32 { DEFAULT_MAX_DB }
fn default_target_range() -> f32 { DEFAULT_TARGET_RANGE }
fn default_sensitivity() -> f32 { DEFAULT_SENSITIVITY }
fn default_raw_gain() -> f32 { DEFAULT_RAW_GAIN }
fn default_response_gamma() -> f32 { DEFAULT_RESPONSE_GAMMA }
fn default_spectrum_tilt() -> f32 { DEFAULT_SPECTRUM_TILT }
fn default_eq_bass() -> f32 { DEFAULT_EQ_BASS }
fn default_eq_mid() -> f32 { DEFAULT_EQ_MID }
fn default_eq_treble() -> f32 { DEFAULT_EQ_TREBLE }
fn default_noise_gate() -> f32 { DEFAULT_NOISE_GATE }
fn default_agc_enabled() -> bool { DEFAULT_AGC_ENABLED }
fn default_agc_attack() -> f32 { DEFAULT_AGC_ATTACK }
fn default_agc_release() -> f32 { DEFAULT_AGC_RELEASE }
fn default_poll_rate_hz() -> f32 { DEFAULT_POLL_RATE }
fn default_attack_ms() -> f32 { DEFAULT_ATTACK_MS }
fn default_decay_ms() -> f32 { DEFAULT_DECAY_MS }
fn default_average_size() -> usize { DEFAULT_AVERAGE_SIZE }
fn default_peak_hold_ms() -> f32 { DEFAULT_PEAK_HOLD_MS }
fn default_gravity_coeff() -> f32 { DEFAULT_GRAVITY_COEFF }
fn default_beat_sensitivity() -> f32 { DEFAULT_BEAT_SENS }
fn default_energy_threshold() -> f32 { DEFAULT_ENERGY_THRESHOLD }
fn default_bpm_phase_smoothing() -> f32 { DEFAULT_BPM_PHASE_SMOOTHING }
fn default_freq_smoothing() -> f32 { DEFAULT_FREQ_SMOOTHING }
fn default_dc_block() -> bool { DEFAULT_DC_BLOCK }
fn default_per_band_ballistics() -> bool { DEFAULT_PER_BAND_BALLISTICS }
fn default_per_frame_normalization() -> bool { DEFAULT_PER_FRAME_NORMALIZATION }
fn default_beat_envelope_decay() -> f32 { DEFAULT_BEAT_ENVELOPE_DECAY_MS }
fn default_harmonic_emphasis() -> bool { DEFAULT_HARMONIC_EMPHASIS }

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            fft_size: DEFAULT_FFT_SIZE, window_function: WindowFunction::default(),
            num_bars: DEFAULT_NUM_BARS, frequency_scaling: FrequencyScaling::default(),
            min_frequency_hz: DEFAULT_MIN_FREQ, max_frequency_hz: DEFAULT_MAX_FREQ,
            channel_mix: ChannelMixMode::default(), weighting: WeightingCurve::default(),
            dc_block: DEFAULT_DC_BLOCK, frequency_smoothing: DEFAULT_FREQ_SMOOTHING,
            harmonic_emphasis: DEFAULT_HARMONIC_EMPHASIS,

            sensitivity: DEFAULT_SENSITIVITY, raw_gain: DEFAULT_RAW_GAIN,
            response_gamma: DEFAULT_RESPONSE_GAMMA, noise_gate: DEFAULT_NOISE_GATE,
            min_decibels: DEFAULT_MIN_DB, max_decibels: DEFAULT_MAX_DB,
            target_dynamic_range_db: DEFAULT_TARGET_RANGE,
            per_frame_normalization: DEFAULT_PER_FRAME_NORMALIZATION,
            agc_enabled: DEFAULT_AGC_ENABLED, agc_attack: DEFAULT_AGC_ATTACK,
            agc_release: DEFAULT_AGC_RELEASE,

            spectrum_tilt: DEFAULT_SPECTRUM_TILT, eq_bass_db: DEFAULT_EQ_BASS,
            eq_mid_db: DEFAULT_EQ_MID, eq_treble_db: DEFAULT_EQ_TREBLE,

            poll_rate_hz: DEFAULT_POLL_RATE, attack_time_ms: DEFAULT_ATTACK_MS,
            decay_time_ms: DEFAULT_DECAY_MS, decay_mode: DecayMode::default(),
            average_size: DEFAULT_AVERAGE_SIZE, peak_hold_ms: DEFAULT_PEAK_HOLD_MS,
            gravity_coefficient: DEFAULT_GRAVITY_COEFF,

            per_band_ballistics: DEFAULT_PER_BAND_BALLISTICS,
            bass_attack_ms: DEFAULT_ATTACK_MS, bass_decay_ms: DEFAULT_DECAY_MS,
            mid_attack_ms: DEFAULT_ATTACK_MS, mid_decay_ms: DEFAULT_DECAY_MS,
            treble_attack_ms: DEFAULT_ATTACK_MS, treble_decay_ms: DEFAULT_DECAY_MS,

            beat_focus: BeatFocusMode::default(),
            beat_detection_sensitivity: DEFAULT_BEAT_SENS,
            energy_threshold_multiplier: DEFAULT_ENERGY_THRESHOLD,
            bpm_lock_enabled: false, bpm_phase_smoothing: DEFAULT_BPM_PHASE_SMOOTHING,
            beat_envelope_decay_ms: DEFAULT_BEAT_ENVELOPE_DECAY_MS,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Audio presets (built-in + custom)
// ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum AudioPreset {
    Universal, Pop, Rock, Electronic, HipHop, Classical, Piano, Jazz, Ambient,
    LoFi, Metal, Country, RnB, Vocal, Podcast,
}

impl AudioPreset {
    pub fn all() -> [AudioPreset; 15] {
        [
            AudioPreset::Universal, AudioPreset::Pop, AudioPreset::Rock,
            AudioPreset::Electronic, AudioPreset::HipHop, AudioPreset::Classical,
            AudioPreset::Piano, AudioPreset::Jazz, AudioPreset::Ambient,
            AudioPreset::LoFi, AudioPreset::Metal, AudioPreset::Country,
            AudioPreset::RnB, AudioPreset::Vocal, AudioPreset::Podcast,
        ]
    }
    pub fn display_name(self) -> &'static str {
        match self {
            AudioPreset::Universal => "Universal",
            AudioPreset::Pop => "Pop",
            AudioPreset::Rock => "Rock",
            AudioPreset::Electronic => "Electronic / EDM",
            AudioPreset::HipHop => "Hip-hop",
            AudioPreset::Classical => "Classical",
            AudioPreset::Piano => "Piano",
            AudioPreset::Jazz => "Jazz",
            AudioPreset::Ambient => "Ambient",
            AudioPreset::LoFi => "Lo-Fi",
            AudioPreset::Metal => "Metal",
            AudioPreset::Country => "Country",
            AudioPreset::RnB => "R&B / Soul",
            AudioPreset::Vocal => "Vocal / Acapella",
            AudioPreset::Podcast => "Podcast / Voice",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            AudioPreset::Universal => "Balanced defaults that work well across most music.",
            AudioPreset::Pop => "Punchy and bright, emphasises vocals and kicks for modern radio pop.",
            AudioPreset::Rock => "Mid-forward with dynamic transients, good for guitars and drums.",
            AudioPreset::Electronic => "Kick-focused with tight decay and BPM lock, tuned for EDM and house.",
            AudioPreset::HipHop => "Bass-heavy and punchy, emphasises 808s and kick drums.",
            AudioPreset::Classical => "Preserves dynamic range (AGC off), softer attack, long decay, clean tones.",
            AudioPreset::Piano => "Clean tone with sustain, good for solo piano and acoustic material.",
            AudioPreset::Jazz => "Warm and smooth, gentler response curve and longer decay.",
            AudioPreset::Ambient => "Slow and flowing, heavy smoothing, soft attack.",
            AudioPreset::LoFi => "Soft, warm, and slow. Long smoothing window and rolled-off highs for a mellow look.",
            AudioPreset::Metal => "Aggressive and punchy. Fast attack, wide dynamic range, mid-forward.",
            AudioPreset::Country => "Clean and dynamic. Balanced mids with moderate smoothing.",
            AudioPreset::RnB => "Smooth bass emphasis with vocal presence. Sweet spot between Pop and Hip-hop.",
            AudioPreset::Vocal => "Mid-focused and voice-oriented. Narrow band, longer decay, gentle comp.",
            AudioPreset::Podcast => "Narrow speech band with heavy smoothing, clean and steady.",
        }
    }
    pub fn settings(self) -> AudioSettings {
        let d = AudioSettings::default();
        match self {
            AudioPreset::Universal => d,

            AudioPreset::Pop => AudioSettings {
                fft_size: 2048, window_function: WindowFunction::Hann,
                sensitivity: 55.0, response_gamma: 1.8, spectrum_tilt: 1.5,
                target_dynamic_range_db: 55.0, decay_time_ms: 200.0,
                decay_mode: DecayMode::Exponential, average_size: 2,
                beat_detection_sensitivity: 0.6, energy_threshold_multiplier: 1.4,
                ..d
            },
            AudioPreset::Rock => AudioSettings {
                fft_size: 2048, window_function: WindowFunction::Hamming,
                sensitivity: 55.0, response_gamma: 1.6, spectrum_tilt: 2.0,
                target_dynamic_range_db: 60.0, attack_time_ms: 3.0,
                decay_time_ms: 250.0, decay_mode: DecayMode::Exponential,
                average_size: 2, beat_detection_sensitivity: 0.55,
                eq_mid_db: 1.5, beat_focus: BeatFocusMode::Mid, ..d
            },
            AudioPreset::Electronic => AudioSettings {
                fft_size: 1024, window_function: WindowFunction::Hann,
                sensitivity: 60.0, response_gamma: 2.0, spectrum_tilt: 1.5,
                target_dynamic_range_db: 50.0, attack_time_ms: 0.0,
                decay_time_ms: 150.0, decay_mode: DecayMode::Gravity,
                average_size: 1, beat_detection_sensitivity: 0.7,
                energy_threshold_multiplier: 1.3, bpm_lock_enabled: true,
                eq_bass_db: 1.5, eq_treble_db: 1.0, beat_focus: BeatFocusMode::Bass, ..d
            },
            AudioPreset::HipHop => AudioSettings {
                fft_size: 1024, window_function: WindowFunction::Hann,
                sensitivity: 65.0, response_gamma: 1.8, spectrum_tilt: 1.0,
                target_dynamic_range_db: 55.0, attack_time_ms: 0.0,
                decay_time_ms: 250.0, decay_mode: DecayMode::Gravity,
                average_size: 1, beat_detection_sensitivity: 0.7,
                energy_threshold_multiplier: 1.3, eq_bass_db: 2.5,
                beat_focus: BeatFocusMode::Bass, ..d
            },
            AudioPreset::Classical => AudioSettings {
                fft_size: 4096, window_function: WindowFunction::BlackmanHarris,
                min_decibels: -95.0, max_decibels: -15.0, target_dynamic_range_db: 75.0,
                sensitivity: 65.0, response_gamma: 1.2, spectrum_tilt: 4.5,
                agc_enabled: false, attack_time_ms: 15.0, decay_time_ms: 500.0,
                decay_mode: DecayMode::Linear, average_size: 3,
                beat_detection_sensitivity: 0.4, max_frequency_hz: 18000.0,
                beat_focus: BeatFocusMode::Mid, ..d
            },
            AudioPreset::Piano => AudioSettings {
                fft_size: 4096, window_function: WindowFunction::BlackmanHarris,
                min_decibels: -90.0, max_decibels: -20.0, target_dynamic_range_db: 65.0,
                sensitivity: 60.0, response_gamma: 1.3, spectrum_tilt: 4.5,
                attack_time_ms: 5.0, decay_time_ms: 600.0, decay_mode: DecayMode::Exponential,
                average_size: 2, beat_detection_sensitivity: 0.35,
                max_frequency_hz: 5000.0, eq_mid_db: 1.0,
                harmonic_emphasis: true,
                beat_focus: BeatFocusMode::Mid, ..d
            },
            AudioPreset::Jazz => AudioSettings {
                fft_size: 2048, window_function: WindowFunction::Hamming,
                min_decibels: -85.0, max_decibels: -22.0, target_dynamic_range_db: 60.0,
                sensitivity: 55.0, response_gamma: 1.4, spectrum_tilt: 3.5,
                attack_time_ms: 10.0, decay_time_ms: 400.0, decay_mode: DecayMode::Exponential,
                average_size: 3, beat_detection_sensitivity: 0.45, ..d
            },
            AudioPreset::Ambient => AudioSettings {
                fft_size: 4096, window_function: WindowFunction::BlackmanHarris,
                min_decibels: -95.0, max_decibels: -25.0, target_dynamic_range_db: 70.0,
                sensitivity: 70.0, response_gamma: 1.2, spectrum_tilt: 4.5,
                attack_time_ms: 40.0, decay_time_ms: 800.0, decay_mode: DecayMode::Exponential,
                average_size: 6, beat_detection_sensitivity: 0.3, ..d
            },

            AudioPreset::LoFi => AudioSettings {
                fft_size: 2048, window_function: WindowFunction::Hann,
                min_decibels: -80.0, max_decibels: -22.0, target_dynamic_range_db: 55.0,
                sensitivity: 60.0, response_gamma: 1.4, spectrum_tilt: 2.0,
                attack_time_ms: 8.0, decay_time_ms: 400.0,
                decay_mode: DecayMode::Exponential,
                average_size: 4, beat_detection_sensitivity: 0.35,
                energy_threshold_multiplier: 1.8, eq_treble_db: -1.0,
                beat_focus: BeatFocusMode::Bass, ..d
            },
            AudioPreset::Metal => AudioSettings {
                fft_size: 2048, window_function: WindowFunction::Hamming,
                min_decibels: -85.0, max_decibels: -18.0, target_dynamic_range_db: 55.0,
                sensitivity: 60.0, response_gamma: 1.9, spectrum_tilt: 1.0,
                attack_time_ms: 1.0, decay_time_ms: 180.0,
                decay_mode: DecayMode::Exponential,
                average_size: 1, beat_detection_sensitivity: 0.7,
                energy_threshold_multiplier: 1.4, eq_mid_db: 2.5, eq_treble_db: 1.0,
                beat_focus: BeatFocusMode::Mid, ..d
            },
            AudioPreset::Country => AudioSettings {
                fft_size: 2048, window_function: WindowFunction::Hann,
                sensitivity: 55.0, response_gamma: 1.4, spectrum_tilt: 3.0,
                attack_time_ms: 6.0, decay_time_ms: 300.0,
                average_size: 2, beat_detection_sensitivity: 0.5,
                eq_mid_db: 1.0, beat_focus: BeatFocusMode::Mid, ..d
            },
            AudioPreset::RnB => AudioSettings {
                fft_size: 2048, window_function: WindowFunction::Hann,
                sensitivity: 58.0, response_gamma: 1.6, spectrum_tilt: 1.5,
                attack_time_ms: 4.0, decay_time_ms: 320.0,
                decay_mode: DecayMode::Exponential,
                average_size: 2, beat_detection_sensitivity: 0.5,
                eq_bass_db: 1.5, eq_mid_db: 1.0,
                beat_focus: BeatFocusMode::Bass, ..d
            },
            AudioPreset::Vocal => AudioSettings {
                fft_size: 4096, window_function: WindowFunction::Hann,
                min_frequency_hz: 80.0, max_frequency_hz: 12000.0,
                min_decibels: -90.0, max_decibels: -20.0, target_dynamic_range_db: 65.0,
                sensitivity: 60.0, response_gamma: 1.3, spectrum_tilt: 3.5,
                attack_time_ms: 10.0, decay_time_ms: 400.0,
                average_size: 3, beat_detection_sensitivity: 0.3,
                eq_bass_db: -3.0, eq_mid_db: 2.0,
                beat_focus: BeatFocusMode::Mid, ..d
            },
            AudioPreset::Podcast => AudioSettings {
                fft_size: 4096, window_function: WindowFunction::BlackmanHarris,
                min_frequency_hz: 100.0, max_frequency_hz: 6000.0,
                min_decibels: -95.0, max_decibels: -20.0, target_dynamic_range_db: 70.0,
                sensitivity: 65.0, response_gamma: 1.2, spectrum_tilt: 4.0,
                attack_time_ms: 20.0, decay_time_ms: 600.0,
                decay_mode: DecayMode::Exponential,
                average_size: 4, beat_detection_sensitivity: 0.25,
                eq_bass_db: -4.0, eq_mid_db: 2.5, eq_treble_db: -1.0,
                beat_focus: BeatFocusMode::Mid, ..d
            },
        }
    }
    pub fn apply(self, s: &mut AudioSettings) { *s = self.settings(); }
    pub fn detect(s: &AudioSettings) -> Option<AudioPreset> {
        for p in Self::all() {
            if p.settings() == *s { return Some(p); }
        }
        None
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CustomAudioPreset {
    pub name: String,
    pub settings: AudioSettings,
}

// ─────────────────────────────────────────────────────────────
// Preset factory helpers
// ─────────────────────────────────────────────────────────────

fn smp(name: &str, colors: Vec<[u8; 3]>, mode: StaticDistributionMode, bars: usize) -> StaticMultiPreset {
    StaticMultiPreset { name: name.into(), colors, mode, bars_per_color: bars, is_premade: true }
}
fn gp(name: &str, stops: Vec<GradientStop>, angle: f32) -> GradientPreset {
    GradientPreset { name: name.into(), stops, angle, is_premade: true }
}
fn gs(color: [u8; 3], position: f32) -> GradientStop { GradientStop { color, position } }
fn ccms(shape: MotionShape, blend: BlendMode, speed: f32, trail: f32, stutter: f32, wave: f32, wipe: f32, off: [u8; 3], off_a: u8) -> MotionSettings {
    MotionSettings {
        shape, blend, speed, trail_length: trail, stutter_intensity: stutter,
        wave_frequency: wave, wipe_softness: wipe,
        off_color: off, off_color_opacity: off_a,
    }
}
fn ccp(name: &str, colors: Vec<[u8; 3]>, speed: f32, motion: MotionSettings) -> ColorChangePreset {
    ColorChangePreset { name: name.into(), colors, speed, is_premade: true, motion }
}

fn default_static_colors() -> Vec<[u8; 3]> { vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]] }
fn default_gradient_stops() -> Vec<GradientStop> {
    vec![GradientStop { color: [255, 0, 0], position: 0.0 }, GradientStop { color: [0, 0, 255], position: 1.0 }]
}
fn default_speed() -> f32 { 0.5 }
fn default_sequential_bars_per_color() -> usize { 1 }
pub fn default_visualizer_outline_thickness() -> f32 { 1.0 }
pub fn default_visualizer_outline_color() -> [u8; 3] { [0, 0, 0] }

fn default_static_single_source() -> ColorSource {
    ColorSource::Dynamic { palette_size: 1, variants_per_color: 0, priority: PalettePriority::MostCommon }
}
fn default_static_multi_source() -> ColorSource { ColorSource::Static(default_static_colors()) }
fn default_animated_source() -> ColorSource { ColorSource::Static(default_static_colors()) }
fn default_gradient_source() -> GradientSource { GradientSource::Static(default_gradient_stops()) }

fn default_color_presets() -> Vec<ColorPreset> {
    let colors = [
        [255, 0, 0], [255, 128, 0], [255, 255, 0], [128, 255, 0], [0, 255, 0], [0, 255, 128],
        [0, 255, 255], [0, 128, 255], [0, 0, 255], [128, 0, 255], [255, 0, 255], [255, 0, 128],
        [255, 255, 255], [200, 200, 200], [150, 150, 150], [100, 100, 100], [50, 50, 50], [0, 0, 0],
        [255, 105, 180], [255, 165, 0], [75, 0, 130], [0, 191, 255], [34, 139, 34], [139, 69, 19]
    ];
    colors.iter().map(|&c| ColorPreset { color: c, is_premade: true }).collect()
}

fn default_static_multi_presets() -> Vec<StaticMultiPreset> {
    use StaticDistributionMode::*;
    vec![
        smp("Sunset Boulevard",  vec![[255, 94, 0], [255, 0, 128], [128, 0, 255]],                              Blocks,     1),
        smp("Deep Ocean",        vec![[0, 30, 60], [0, 100, 160], [0, 180, 216], [144, 224, 239]],             Sequential, 2),
        smp("Enchanted Forest",  vec![[20, 80, 20], [34, 139, 34], [50, 205, 50], [144, 238, 144]],            Random,     1),
        smp("Cyberpunk Neon",    vec![[255, 0, 255], [0, 255, 255], [255, 255, 0]],                           Vertical,   1),
        smp("Traffic Light",     vec![[255, 60, 60], [255, 200, 60], [60, 220, 60]],                          Vertical,   1),
        smp("Thermal Camera",    vec![[40, 20, 80], [140, 20, 140], [240, 80, 40], [255, 220, 60]],            Vertical,   1),
        smp("Fire & Ice",        vec![[255, 50, 0], [255, 150, 0], [100, 200, 255], [0, 150, 255]],            Vertical,   1),
        smp("Vaporwave",         vec![[255, 113, 206], [1, 205, 254], [5, 255, 161], [185, 103, 255]],          Vertical,   1),
        smp("Rainbow Blocks",    vec![[255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [75, 0, 130], [148, 0, 211]], Blocks, 1),
        smp("Candy Shop",        vec![[255, 105, 180], [255, 215, 0], [0, 255, 255]],                          Sequential, 3),
        smp("Lava Lamp",         vec![[255, 0, 0], [255, 100, 0], [150, 0, 200], [80, 0, 150]],                Random,     1),
        smp("Monochrome",        vec![[255, 255, 255], [150, 150, 150], [80, 80, 80]],                        Sequential, 2),
        smp("Iris Stack",        vec![[80, 40, 20], [180, 80, 60], [240, 180, 100], [220, 240, 180], [100, 180, 120]], Vertical, 1),
        smp("Pastel Dream",      vec![[255, 182, 193], [173, 216, 230], [144, 238, 144], [255, 255, 224]],     Blocks,     1),
        smp("Tropical",          vec![[0, 200, 150], [255, 200, 0], [255, 80, 80]],                           Random,     1),
        smp("Northern Sky",      vec![[10, 10, 60], [40, 40, 140], [80, 120, 220], [160, 200, 255]],           Vertical,   1),
        smp("Galactic Core",     vec![[255, 255, 255], [255, 230, 150], [200, 100, 255], [60, 0, 120]],        Vertical,   1),
        smp("Autumn Leaves",     vec![[100, 30, 0], [200, 80, 20], [230, 150, 40], [255, 220, 100]],           Vertical,   1),
        smp("Retro Arcade",      vec![[0, 255, 0], [0, 100, 0], [255, 0, 255], [100, 0, 100]],                 Sequential, 4),
        smp("Bioluminescence",   vec![[0, 50, 100], [0, 150, 200], [0, 255, 200], [100, 255, 150]],            Vertical,   1),
        smp("Candy Cane",        vec![[255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 255]],             Sequential, 1),
        smp("Midnight Sun",      vec![[20, 0, 40], [80, 0, 80], [200, 0, 100], [255, 100, 0]],                 Vertical,   1),
    ]
}

fn default_gradient_presets() -> Vec<GradientPreset> {
    vec![
        gp("Sunset",           vec![gs([255, 94, 0], 0.0), gs([255, 0, 128], 0.5), gs([128, 0, 255], 1.0)], 0.0),
        gp("Ocean Deep",       vec![gs([0, 30, 60], 0.0), gs([0, 180, 216], 0.5), gs([144, 224, 239], 1.0)], 90.0),
        gp("Rainbow",          vec![gs([255, 0, 0], 0.0), gs([255, 127, 0], 0.17), gs([255, 255, 0], 0.33), gs([0, 255, 0], 0.5), gs([0, 0, 255], 0.67), gs([75, 0, 130], 0.83), gs([148, 0, 211], 1.0)], 45.0),
        gp("Fire",             vec![gs([255, 0, 0], 0.0), gs([255, 165, 0], 0.5), gs([255, 255, 0], 1.0)], 90.0),
        gp("Aurora",           vec![gs([0, 255, 150], 0.0), gs([0, 150, 255], 0.5), gs([150, 0, 255], 1.0)], 90.0),
        gp("Cotton Candy",     vec![gs([255, 182, 193], 0.0), gs([173, 216, 230], 0.5), gs([255, 105, 180], 1.0)], 0.0),
        gp("Gold Rush",        vec![gs([80, 40, 0], 0.0), gs([255, 180, 0], 0.5), gs([255, 255, 150], 1.0)], 90.0),
        gp("Neon Nights",      vec![gs([255, 0, 150], 0.0), gs([100, 0, 255], 0.5), gs([0, 255, 255], 1.0)], 45.0),
        gp("Enchanted Forest", vec![gs([0, 50, 0], 0.0), gs([50, 150, 50], 0.5), gs([200, 255, 150], 1.0)], 90.0),
        gp("Grayscale",        vec![gs([0, 0, 0], 0.0), gs([255, 255, 255], 1.0)], 90.0),
        gp("Twilight",         vec![gs([25, 25, 80], 0.0), gs([150, 50, 150], 0.5), gs([255, 150, 100], 1.0)], 90.0),
        gp("Ice",              vec![gs([200, 240, 255], 0.0), gs([100, 180, 255], 0.5), gs([30, 100, 200], 1.0)], 90.0),
        gp("Lava Flow",        vec![gs([10, 0, 0], 0.0), gs([180, 30, 0], 0.3), gs([255, 180, 0], 0.7), gs([255, 255, 200], 1.0)], 90.0),
        gp("Cyber Grid",       vec![gs([255, 0, 150], 0.0), gs([100, 0, 255], 0.5), gs([0, 255, 255], 1.0)], 45.0),
        gp("Cherry Blossom",   vec![gs([255, 200, 220], 0.0), gs([255, 105, 180], 0.5), gs([180, 50, 120], 1.0)], 0.0),
        gp("Northern Lights",  vec![gs([0, 20, 20], 0.0), gs([0, 180, 150], 0.4), gs([100, 255, 180], 0.7), gs([200, 240, 255], 1.0)], 90.0),
        gp("Synthwave",        vec![gs([20, 0, 50], 0.0), gs([255, 0, 100], 0.5), gs([255, 200, 0], 1.0)], 0.0),
        gp("Deep Space",       vec![gs([0, 0, 0], 0.0), gs([20, 20, 60], 0.4), gs([80, 60, 150], 0.8), gs([255, 255, 255], 1.0)], 90.0),
        gp("Radioactive",      vec![gs([10, 20, 10], 0.0), gs([50, 255, 50], 0.5), gs([200, 255, 200], 1.0)], 90.0),
        gp("Desert Mirage",    vec![gs([255, 230, 200], 0.0), gs([240, 160, 80], 0.4), gs([180, 80, 30], 1.0)], 0.0),
        gp("Electric Blue",    vec![gs([0, 50, 255], 0.0), gs([0, 200, 255], 0.5), gs([255, 255, 255], 1.0)], 90.0),
        gp("Vintage Sunset",   vec![gs([50, 20, 30], 0.0), gs([200, 80, 60], 0.5), gs([250, 200, 100], 1.0)], 90.0),
    ]
}

fn default_color_change_presets() -> Vec<ColorChangePreset> {
    use MotionShape::*; use BlendMode::*;
    vec![
        ccp("Rainbow Cycle",     vec![[255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0], [0, 0, 255], [75, 0, 130], [148, 0, 211]],
            0.5, ccms(Uniform, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Aurora Veil",       vec![[0, 255, 180], [60, 140, 255], [180, 60, 255]],
            0.35, ccms(VerticalWipe, Smooth, 1.0, 0.25, 0.3, 2.0, 0.35, [4, 8, 20], 220)),
        ccp("Comet Tail",        vec![[255, 200, 80], [255, 90, 0], [120, 20, 0]],
            0.55, ccms(Comet, Smooth, 1.0, 0.55, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Neon Rain",         vec![[255, 0, 255], [0, 255, 255], [0, 255, 120]],
            0.75, ccms(Chase, Hard, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Heartbeat Pop",     vec![[255, 60, 90], [255, 180, 180]],
            0.45, ccms(Heartbeat, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [30, 0, 10], 180)),
        ccp("Candy Sparkle",     vec![[255, 105, 180], [255, 215, 0], [0, 255, 255]],
            0.4, ccms(Sparkle, Hard, 1.0, 0.25, 0.5, 2.0, 0.1, [20, 0, 20], 200)),
        ccp("Fireflies",         vec![[255, 220, 0], [255, 120, 0]],
            0.5, ccms(Confetti, Smooth, 1.0, 0.25, 0.35, 2.0, 0.1, [10, 5, 0], 150)),
        ccp("Diagonal Storm",    vec![[40, 40, 255], [200, 0, 255], [255, 0, 80]],
            0.6, ccms(DiagonalWipe, Smooth, 1.0, 0.25, 0.3, 2.0, 0.2, [0, 0, 10], 230)),
        ccp("Plasma Drift",      vec![[255, 50, 200], [0, 200, 255], [180, 255, 0]],
            0.3, ccms(Plasma, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Electric Pulse",    vec![[0, 255, 255], [255, 0, 255]],
            0.65, ccms(Pulse, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Bouncing Comet",    vec![[255, 255, 0], [255, 0, 0]],
            0.5, ccms(BouncingBall, Smooth, 1.0, 0.4, 0.3, 2.0, 0.1, [10, 10, 0], 220)),
        ccp("Retro Strobe",      vec![[255, 80, 80], [80, 80, 255]],
            0.85, ccms(Strobe, Hard, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Smooth Sinelon",    vec![[120, 255, 120], [0, 180, 255], [255, 120, 255]],
            0.4, ccms(Sinelon, Smooth, 1.0, 0.6, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Radial Burst",      vec![[255, 200, 0], [255, 50, 100], [100, 50, 255]],
            0.4, ccms(RadialWipe, Smooth, 1.0, 0.25, 0.3, 2.0, 0.15, [0, 0, 0], 255)),
        ccp("Spiral Galaxy",     vec![[255, 255, 255], [150, 100, 255], [50, 20, 150]],
            0.3, ccms(Spiral, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Ripple Pond",       vec![[0, 150, 200], [100, 220, 255], [255, 255, 255]],
            0.35, ccms(Ripple, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Wave Interference", vec![[255, 100, 100], [100, 255, 100], [100, 100, 255]],
            0.5, ccms(WaveInterference, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Cyber Rain",        vec![[0, 255, 100], [0, 150, 50], [0, 50, 20]],
            0.6, ccms(DigitalRain, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Meteor Strike",     vec![[255, 255, 255], [255, 100, 0], [100, 0, 0]],
            0.7, ccms(MeteorShower, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Zen Breath",        vec![[100, 200, 200], [150, 255, 150]],
            0.2, ccms(Breathing, Smooth, 1.0, 0.25, 0.3, 2.0, 0.1, [10, 20, 20], 200)),
        ccp("Hyper Flash",       vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            1.0, ccms(FlipFlop, Hard, 1.0, 0.25, 0.3, 2.0, 0.1, [0, 0, 0], 255)),
        ccp("Kaleidoscope",      vec![[255, 100, 200], [100, 255, 200], [200, 100, 255]],
            0.4, ccms(Kaleidoscope, Smooth, 1.0, 0.25, 0.3, 3.0, 0.1, [0, 0, 0], 255)),
    ]
}

fn deserialize_color_presets<'de, D>(deserializer: D) -> Result<Vec<ColorPreset>, D::Error>
where D: serde::Deserializer<'de> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Helper {
        Old([u8; 3]),
        New(ColorPreset),
    }
    let vals: Vec<Helper> = Vec::deserialize(deserializer)?;
    Ok(vals.into_iter().map(|v| match v {
        Helper::Old(c) => ColorPreset { color: c, is_premade: false },
        Helper::New(p) => p,
    }).collect())
}

// ─────────────────────────────────────────────────────────────
// App config
// ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
pub struct AppConfig {
    // Layout
    pub bar_width: f32, pub bar_gap: f32, pub bar_radius: f32,
    pub bg_opacity: u8,
    #[serde(default)] pub bg_color: [u8; 3],
    pub x_offset: f32, pub y_offset: f32,
    pub max_bar_height: f32, pub show_fps: bool,
    #[serde(default = "default_height_scale")] pub height_scale: f32,

    // Opacity / menu
    #[serde(default = "default_visualizer_opacity")] pub visualizer_opacity: u8,
    #[serde(default = "default_settings_scale")] pub settings_scale: f32,
    #[serde(default = "default_open_settings_on_launch")] pub open_settings_on_launch: bool,
    #[serde(default = "default_show_help_text")] pub show_help_text: bool,

    // Outlines
    #[serde(default, alias = "bar_outline_enabled")]   pub visualizer_outline_enabled: bool,
    #[serde(default = "default_visualizer_outline_thickness", alias = "bar_outline_thickness")] pub visualizer_outline_thickness: f32,
    #[serde(default = "default_visualizer_outline_color", alias = "bar_outline_color")]        pub visualizer_outline_color: [u8; 3],

    // Visualizer behavior
    #[serde(default)] pub visualizer_direction: VisualizerDirection,
    #[serde(default)] pub frequency_direction: FrequencyDirection,
    #[serde(default)] pub color_mode: ColorMode,
    #[serde(default)] pub bar_interpolation: BarInterpolation,
    #[serde(default = "default_static_single_source")] pub static_single_source: ColorSource,
    #[serde(default = "default_static_multi_source")]  pub static_multi_source: ColorSource,
    #[serde(default)] pub static_mode: StaticDistributionMode,
    #[serde(default = "default_sequential_bars_per_color")] pub sequential_bars_per_color: usize,
    #[serde(default = "default_static_multi_presets")] pub static_multi_presets: Vec<StaticMultiPreset>,
    #[serde(default)] pub vertical_color_style: VerticalColorStyle,

    // Gradients
    #[serde(default = "default_gradient_source")] pub gradient_source: GradientSource,
    #[serde(default)] pub gradient_angle: f32,
    #[serde(default = "default_gradient_presets")] pub gradient_presets: Vec<GradientPreset>,

    // Animated
    #[serde(default = "default_animated_source")] pub animated_source: ColorSource,
    #[serde(default = "default_speed")] pub color_change_speed: f32,
    #[serde(default = "default_color_change_presets")] pub color_change_presets: Vec<ColorChangePreset>,
    #[serde(default)] pub color_change_motion: MotionSettings,

    // Presets
    #[serde(default = "default_color_presets", deserialize_with = "deserialize_color_presets")] pub color_presets: Vec<ColorPreset>,

    // Track info / audio
    #[serde(default)] pub track_info: TrackInfoConfig,
    #[serde(default)] pub audio: AudioSettings,
    #[serde(default)] pub custom_audio_presets: Vec<CustomAudioPreset>,

    // Legacy migration fields (kept for old configs; consumed during load())
    #[serde(default, skip_serializing_if = "Option::is_none")] pub num_bars: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub fft_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub audio_update_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub sensitivity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub bar_color: Option<[u8; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub static_colors: Option<Vec<[u8; 3]>>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub gradient_stops: Option<Vec<GradientStop>>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub color_change_colors: Option<Vec<[u8; 3]>>,
}

fn default_height_scale() -> f32 { DEFAULT_HEIGHT_SCALE }
fn default_visualizer_opacity() -> u8 { DEFAULT_VISUALIZER_OPACITY }
fn default_settings_scale() -> f32 { DEFAULT_SETTINGS_SCALE }
fn default_open_settings_on_launch() -> bool { DEFAULT_OPEN_SETTINGS_ON_LAUNCH }
fn default_show_help_text() -> bool { DEFAULT_SHOW_HELP_TEXT }

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            bar_width: DEFAULT_BAR_WIDTH, bar_gap: DEFAULT_BAR_GAP, bar_radius: DEFAULT_BAR_RADIUS,
            bg_opacity: DEFAULT_BG_OPACITY, bg_color: [0, 0, 0],
            x_offset: DEFAULT_X_OFFSET, y_offset: DEFAULT_Y_OFFSET,
            max_bar_height: DEFAULT_MAX_BAR_HEIGHT,
            show_fps: false,
            height_scale: DEFAULT_HEIGHT_SCALE,
            visualizer_opacity: DEFAULT_VISUALIZER_OPACITY,
            settings_scale: DEFAULT_SETTINGS_SCALE,
            open_settings_on_launch: DEFAULT_OPEN_SETTINGS_ON_LAUNCH,
            show_help_text: DEFAULT_SHOW_HELP_TEXT,
            visualizer_outline_enabled: false,
            visualizer_outline_thickness: default_visualizer_outline_thickness(),
            visualizer_outline_color: default_visualizer_outline_color(),
            visualizer_direction: VisualizerDirection::default(),
            frequency_direction: FrequencyDirection::default(),
            color_mode: ColorMode::StaticSingle,
            bar_interpolation: BarInterpolation::default(),
            static_single_source: default_static_single_source(),
            static_multi_source: default_static_multi_source(),
            static_mode: StaticDistributionMode::Blocks,
            sequential_bars_per_color: default_sequential_bars_per_color(),
            static_multi_presets: default_static_multi_presets(),
            vertical_color_style: VerticalColorStyle::default(),
            gradient_source: default_gradient_source(),
            gradient_angle: 0.0,
            gradient_presets: default_gradient_presets(),
            animated_source: default_animated_source(),
            color_change_speed: default_speed(),
            color_change_presets: default_color_change_presets(),
            color_change_motion: MotionSettings::default(),
            color_presets: default_color_presets(),
            track_info: TrackInfoConfig::default(),
            audio: AudioSettings::default(),
            custom_audio_presets: Vec::new(),
            num_bars: None, fft_size: None, audio_update_ms: None, sensitivity: None,
            bar_color: None, static_colors: None, gradient_stops: None, color_change_colors: None,
        }
    }
}

impl AppConfig {
    pub fn get_path() -> PathBuf {
        dirs::config_dir().unwrap_or_else(|| PathBuf::from("."))
            .join("RustVisualizer").join("config.toml")
    }

    fn refresh_premade_presets(&mut self) {
        let has_any_premade = self.static_multi_presets.iter().any(|p| p.is_premade)
            || self.gradient_presets.iter().any(|p| p.is_premade)
            || self.color_change_presets.iter().any(|p| p.is_premade)
            || self.color_presets.iter().any(|p| p.is_premade);

        if !has_any_premade {
            self.color_presets = default_color_presets();
            self.static_multi_presets = default_static_multi_presets();
            self.gradient_presets = default_gradient_presets();
            self.color_change_presets = default_color_change_presets();
            return;
        }

        self.color_presets.retain(|p| !p.is_premade);
        self.static_multi_presets.retain(|p| !p.is_premade);
        self.gradient_presets.retain(|p| !p.is_premade);
        self.color_change_presets.retain(|p| !p.is_premade);

        let mut new_cp = default_color_presets();
        new_cp.append(&mut self.color_presets);
        self.color_presets = new_cp;

        let mut new_smp = default_static_multi_presets();
        new_smp.append(&mut self.static_multi_presets);
        self.static_multi_presets = new_smp;

        let mut new_gp = default_gradient_presets();
        new_gp.append(&mut self.gradient_presets);
        self.gradient_presets = new_gp;

        let mut new_ccp = default_color_change_presets();
        new_ccp.append(&mut self.color_change_presets);
        self.color_change_presets = new_ccp;
    }

    pub fn load() -> Self {
        let path = Self::get_path();
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(mut config) = toml::from_str::<AppConfig>(&content) {
                if let Some(c) = config.bar_color.take() {
                    config.static_single_source = ColorSource::Static(vec![c]);
                }
                if let Some(c) = config.static_colors.take() {
                    config.static_multi_source = ColorSource::Static(c);
                }
                if let Some(s) = config.gradient_stops.take() {
                    config.gradient_source = GradientSource::Static(s);
                }
                if let Some(c) = config.color_change_colors.take() {
                    config.animated_source = ColorSource::Static(c);
                }
                if let Some(n) = config.num_bars.take() { config.audio.num_bars = n; }
                if let Some(f) = config.fft_size.take() { config.audio.fft_size = f; }
                if let Some(ms) = config.audio_update_ms.take() {
                    if ms > 0 { config.audio.poll_rate_hz = 1000.0 / ms as f32; }
                }
                if let Some(s) = config.sensitivity.take() {
                    config.audio.sensitivity = s.clamp(0.0, 100.0);
                }

                config.refresh_premade_presets();
                return config;
            }
        }
        Self::default()
    }

    pub fn save(&self) {
        let path = Self::get_path();
        if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
        if let Ok(s) = toml::to_string(self) { let _ = std::fs::write(path, s); }
    }
}
