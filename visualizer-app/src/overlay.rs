//! Win32 glue for the single overlay window.
//!
//! Goal: one transparent window that lives *on the desktop* with this z-order
//! (bottom -> top):
//!
//!     wallpaper  <  desktop icons  <  visualizer + settings (this window)  <  every other app
//!
//! How it works
//! ------------
//! * The desktop icons live in a `SHELLDLL_DefView` child of a top-level
//!   "icon host" window (`Progman` or a `WorkerW`, depending on the Windows
//!   version).  We insert our window *directly above that host* in the
//!   top-level z-order.  Normal windows are always above the desktop, so they
//!   stay above us, and the icons stay below us.
//! * Windows re-orders windows when they are activated (e.g. when you click the
//!   settings panel).  A window subclass rewrites `WM_WINDOWPOSCHANGING` so any
//!   z-order change of *our* window is redirected to "just above the icon host".
//! * "Show desktop" can raise the icon host above us, so `Overlay::tick`
//!   periodically checks and re-pins.
//! * The window gets `WS_EX_TOOLWINDOW` (no taskbar button / Alt-Tab entry) and
//!   has `WS_EX_APPWINDOW` cleared.  Winit rewrites the extended style whenever
//!   the click-through flag is toggled, so the flags are re-applied from
//!   `WM_STYLECHANGING`.
//! * DWM composition is explicitly configured to keep the window transparent
//!   even when it gains focus (which normally paints the window background black).

use eframe::egui;
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows_sys::Win32::Graphics::Dwm::{
    DwmExtendFrameIntoClientArea, DwmSetWindowAttribute, MARGINS,
    DWMWA_NCRENDERING_POLICY, DWMWA_TRANSITIONS_FORCEDISABLED, DWMNCRP_DISABLED,
};
use windows_sys::Win32::Graphics::Gdi::ScreenToClient;
use windows_sys::Win32::System::Threading::GetCurrentProcessId;
use windows_sys::Win32::UI::Shell::{DefSubclassProc, SetWindowSubclass};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    EnumWindows, FindWindowExW, GetCursorPos, GetWindow, GetWindowLongPtrW, GetWindowTextW,
    GetWindowThreadProcessId, IsWindow, IsWindowVisible, SetWindowLongPtrW, SetWindowPos,
    ShowWindow, GWL_EXSTYLE, GW_HWNDPREV, STYLESTRUCT, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE,
    SWP_NOZORDER, SW_HIDE, SW_SHOWNA, WINDOWPOS, WM_ERASEBKGND, WM_NCACTIVATE, WM_NCPAINT,
    WM_STYLECHANGING, WM_WINDOWPOSCHANGING, WS_EX_APPWINDOW, WS_EX_LAYERED, WS_EX_TOOLWINDOW,
};

/// Hide the overlay from the taskbar / Alt-Tab (also keeps it from being
/// minimised by "Show desktop").  Set to `false` to keep the old behaviour.
const HIDE_FROM_TASKBAR: bool = true;

/// Unique id for our window subclass (winit uses its own ids).
const SUBCLASS_ID: usize = 0x5649_5A31;

/// Icon-host window the subclass pins us above (0 = unknown).
static ICON_HOST: AtomicIsize = AtomicIsize::new(0);

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

// ───────────────────────── window discovery ─────────────────────────

struct FindOwn {
    pid: u32,
    title: Vec<u16>,
    found: isize,
}

unsafe extern "system" fn enum_own(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let ctx = &mut *(lparam as *mut FindOwn);
    let mut pid = 0u32;
    GetWindowThreadProcessId(hwnd, &mut pid);
    if pid != ctx.pid {
        return 1;
    }
    let mut buf = [0u16; 256];
    let n = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
    if n > 0 && buf[..n as usize] == ctx.title[..] {
        ctx.found = hwnd as isize;
        return 0; // stop
    }
    1
}

fn find_own_window(title: &str) -> Option<isize> {
    let mut ctx = FindOwn {
        pid: unsafe { GetCurrentProcessId() },
        title: title.encode_utf16().collect(),
        found: 0,
    };
    unsafe { EnumWindows(Some(enum_own), &mut ctx as *mut FindOwn as LPARAM) };
    (ctx.found != 0).then_some(ctx.found)
}

unsafe extern "system" fn enum_icon_host(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let out = &mut *(lparam as *mut isize);
    let class = wide("SHELLDLL_DefView");
    let child = FindWindowExW(hwnd, 0 as _, class.as_ptr(), std::ptr::null());
    if child as isize != 0 {
        *out = hwnd as isize;
        return 0; // stop
    }
    1
}

/// Top-level window that hosts the desktop icons (Progman or a WorkerW).
fn find_icon_host() -> Option<isize> {
    let mut out: isize = 0;
    unsafe { EnumWindows(Some(enum_icon_host), &mut out as *mut isize as LPARAM) };
    (out != 0).then_some(out)
}

// ───────────────────────── z-order ─────────────────────────

/// Put `hwnd` directly above `host` (without activating it).
unsafe fn pin_above(hwnd: isize, host: isize) {
    if host == 0 {
        return;
    }
    SetWindowPos(
        hwnd as HWND,
        host as HWND,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
    );
}

/// Is `hwnd` somewhere above `host` in the top-level z-order?
unsafe fn is_above(hwnd: isize, host: isize) -> bool {
    let mut w = GetWindow(host as HWND, GW_HWNDPREV);
    let mut guard = 0;
    while w as isize != 0 && guard < 8192 {
        if w as isize == hwnd {
            return true;
        }
        w = GetWindow(w, GW_HWNDPREV);
        guard += 1;
    }
    false
}

unsafe extern "system" fn subclass_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _id: usize,
    _ref_data: usize,
) -> LRESULT {
    match msg {
        // Prevent the OS from painting the background (black) on focus or resize.
        WM_ERASEBKGND => {
            return 1;
        }
        // Prevent Windows from drawing the non-client area when activated.
        WM_NCACTIVATE => {
            return 1;
        }
        // Prevent Windows from painting the non-client border.
        WM_NCPAINT => {
            return 0;
        }
        // Whenever something (activation, click, ...) tries to re-order *our*
        // window, redirect it to "directly above the desktop icons".
        WM_WINDOWPOSCHANGING => {
            let wp = &mut *(lparam as *mut WINDOWPOS);
            let host = ICON_HOST.load(Ordering::Relaxed);
            if host != 0 && (wp.flags & SWP_NOZORDER) == 0 && wp.hwndInsertAfter as isize != host {
                wp.hwndInsertAfter = host as HWND;
            }
        }
        // Keep WS_EX_TOOLWINDOW and WS_EX_LAYERED even if winit rewrites the style.
        WM_STYLECHANGING if (wparam as i32) == GWL_EXSTYLE => {
            let ss = &mut *(lparam as *mut STYLESTRUCT);
            if HIDE_FROM_TASKBAR {
                ss.styleNew |= WS_EX_TOOLWINDOW;
                ss.styleNew &= !WS_EX_APPWINDOW;
            }
            // Force layered to ensure DWM transparency persists.
            ss.styleNew |= WS_EX_LAYERED;
        }
        _ => {}
    }
    DefSubclassProc(hwnd, msg, wparam, lparam)
}

// ───────────────────────── public API ─────────────────────────

pub struct Overlay {
    hwnd: isize,
    host: isize,
    last_tick: Instant,
}

impl Overlay {
    /// Find our own window by title and install the z-order hook.
    /// Returns `None` if the window does not exist yet (call again next frame).
    pub fn attach(title: &str) -> Option<Self> {
        let hwnd = find_own_window(title)?;
        let host = find_icon_host().unwrap_or(0);
        ICON_HOST.store(host, Ordering::Relaxed);

        unsafe {
            SetWindowSubclass(hwnd as HWND, Some(subclass_proc), SUBCLASS_ID, 0);

            // Fix for black background on focus: extend DWM frame into the whole client area
            let margins = MARGINS { cxLeftWidth: -1, cxRightWidth: -1, cyTopHeight: -1, cyBottomHeight: -1 };
            DwmExtendFrameIntoClientArea(hwnd as HWND, &margins);

            // Disable DWM transitions (which cause flickering/black backgrounds on focus)
            let mut disabled: i32 = 1;
            DwmSetWindowAttribute(
                hwnd as HWND,
                DWMWA_TRANSITIONS_FORCEDISABLED as u32,
                &mut disabled as *mut _ as _,
                std::mem::size_of::<i32>() as u32,
            );

            // Disable non-client rendering entirely
            let mut nc_policy: i32 = DWMNCRP_DISABLED as i32;
            DwmSetWindowAttribute(
                hwnd as HWND,
                DWMWA_NCRENDERING_POLICY as u32,
                &mut nc_policy as *mut _ as _,
                std::mem::size_of::<i32>() as u32,
            );

            if HIDE_FROM_TASKBAR {
                // Hide/show so the shell drops the existing taskbar button.
                let was_visible = IsWindowVisible(hwnd as HWND) != 0;
                if was_visible {
                    ShowWindow(hwnd as HWND, SW_HIDE);
                }
                let ex = GetWindowLongPtrW(hwnd as HWND, GWL_EXSTYLE) as isize;
                let new_ex = (ex | WS_EX_TOOLWINDOW as isize | WS_EX_LAYERED as isize)
                    & !(WS_EX_APPWINDOW as isize);
                SetWindowLongPtrW(hwnd as HWND, GWL_EXSTYLE, new_ex as _);
                if was_visible {
                    ShowWindow(hwnd as HWND, SW_SHOWNA);
                }
            }

            pin_above(hwnd, host);
        }

        Some(Self { hwnd, host, last_tick: Instant::now() })
    }

    pub fn hwnd(&self) -> isize {
        self.hwnd
    }

    /// Cursor position in client (window-local) pixels.
    pub fn cursor_client(&self) -> Option<(f32, f32)> {
        cursor_client(self.hwnd)
    }

    /// Cheap periodic check (self-throttled): repairs the z-order after
    /// "Show desktop" / explorer restarts.
    pub fn tick(&mut self) {
        if self.last_tick.elapsed() < Duration::from_millis(250) {
            return;
        }
        self.last_tick = Instant::now();
        unsafe {
            if self.host == 0 || IsWindow(self.host as HWND) == 0 {
                self.host = find_icon_host().unwrap_or(0);
                ICON_HOST.store(self.host, Ordering::Relaxed);
                if self.host != 0 {
                    pin_above(self.hwnd, self.host);
                }
                return;
            }
            if !is_above(self.hwnd, self.host) {
                pin_above(self.hwnd, self.host);
            }
        }
    }
}

fn cursor_client(hwnd: isize) -> Option<(f32, f32)> {
    unsafe {
        let mut p = POINT { x: 0, y: 0 };
        if GetCursorPos(&mut p) == 0 {
            return None;
        }
        if ScreenToClient(hwnd as HWND, &mut p) == 0 {
            return None;
        }
        Some((p.x as f32, p.y as f32))
    }
}

/// While the overlay is click-through it receives no mouse events, so egui
/// cannot tell when the cursor enters the settings panel.  This thread polls
/// the cursor and wakes the UI when "cursor inside the panel" changes, so the
/// main loop can switch click-through off/on.
///
/// `hot` is the panel rect in client pixels (`None` = settings closed).
/// `cursor_inside` is the last value the UI loop acted on.
pub fn spawn_cursor_watcher(
    ctx: egui::Context,
    hwnd: isize,
    hot: Arc<Mutex<Option<[f32; 4]>>>,
    cursor_inside: Arc<AtomicBool>,
    quit: Arc<AtomicBool>,
) {
    let _ = std::thread::Builder::new()
        .name("overlay-cursor-watch".into())
        .spawn(move || {
            while !quit.load(Ordering::Relaxed) {
                let rect = *hot.lock().unwrap_or_else(|e| e.into_inner());
                match rect {
                    None => std::thread::sleep(Duration::from_millis(100)),
                    Some(r) => {
                        std::thread::sleep(Duration::from_millis(16));
                        let inside = cursor_client(hwnd)
                            .map(|(x, y)| x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3])
                            .unwrap_or(false);
                        if inside != cursor_inside.load(Ordering::Relaxed) {
                            ctx.request_repaint();
                        }
                    }
                }
            }
        });
}
