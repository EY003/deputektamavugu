use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;

use windows::Foundation::TypedEventHandler;
use windows::Media::Control::{
    GlobalSystemMediaTransportControlsSession,
    GlobalSystemMediaTransportControlsSessionManager,
    GlobalSystemMediaTransportControlsSessionMediaProperties,
};
use windows::Storage::Streams::DataReader;
use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, TranslateMessage, MSG, WM_QUIT,
};

#[derive(Clone, Debug)]
pub struct TrackInfo {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub cover_art: Option<Vec<u8>>,
}

pub fn log_smtc(_msg: &str) {}

type UpdateFn = Arc<dyn Fn() + Send + Sync>;

pub fn spawn_smtc_listener<F>(
    tx: Sender<TrackInfo>,
    should_quit: Arc<AtomicBool>,
    session_count: Arc<AtomicUsize>,
    on_update: F,
) where
    F: Fn() + Send + Sync + 'static,
{
    std::thread::spawn(move || {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        let manager = loop {
            if should_quit.load(Ordering::Relaxed) {
                return;
            }
            match GlobalSystemMediaTransportControlsSessionManager::RequestAsync()
                .and_then(|op| op.get())
            {
                Ok(m) => break m,
                Err(_) => std::thread::sleep(std::time::Duration::from_millis(1000)),
            }
        };

        let tx: Arc<Sender<TrackInfo>> = Arc::new(tx);
        let on_update: UpdateFn = Arc::new(on_update);
        let should_quit_cb = Arc::clone(&should_quit);
        let session_count_cb = Arc::clone(&session_count);

        // ── Register CurrentSessionChanged ─────────────────────
        {
            let manager_for_handler = manager.clone();
            let tx_cb = Arc::clone(&tx);
            let on_update_cb = Arc::clone(&on_update);
            let should_quit = Arc::clone(&should_quit_cb);
            let session_count = Arc::clone(&session_count_cb);

            let handler = TypedEventHandler::new(
                move |_sender: &Option<GlobalSystemMediaTransportControlsSessionManager>,
                      _args: &Option<windows::Media::Control::CurrentSessionChangedEventArgs>| {
                    if should_quit.load(Ordering::Relaxed) {
                        return Ok(());
                    }

                    match manager_for_handler.GetCurrentSession() {
                        Ok(session) => {
                            attach_session(&session, &tx_cb, &on_update_cb);
                        }
                        Err(_) => {
                            let _ = tx_cb.send(TrackInfo {
                                title: String::new(),
                                artist: String::new(),
                                album: String::new(),
                                cover_art: None,
                            });
                            on_update_cb();
                        }
                    }

                    if let Ok(sessions) = manager_for_handler.GetSessions() {
                        session_count
                            .store(sessions.Size().unwrap_or(0) as usize, Ordering::Relaxed);
                    }
                    Ok(())
                },
            );

            let _ = manager.CurrentSessionChanged(&handler);
        }

        // ── Initial fetch (music may already be playing) ───────
        if let Ok(session) = manager.GetCurrentSession() {
            attach_session(&session, &tx, &on_update);
        }
        if let Ok(sessions) = manager.GetSessions() {
            session_count.store(sessions.Size().unwrap_or(0) as usize, Ordering::Relaxed);
        }

        // ── Message pump ───────────────────────────────────────
        let mut msg = MSG::default();
        loop {
            if should_quit.load(Ordering::Relaxed) {
                break;
            }
            let ret = unsafe { GetMessageW(&mut msg, None, 0, 0) };
            if ret.0 == 0 || msg.message == WM_QUIT {
                break;
            }
            unsafe {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    });
}

/// Fetch current properties, then register a `MediaPropertiesChanged` handler
/// on the session so future changes are event-driven.
fn attach_session(
    session: &GlobalSystemMediaTransportControlsSession,
    tx: &Arc<Sender<TrackInfo>>,
    on_update: &UpdateFn,
) {
    // Fetch once immediately.
    emit_props(session, tx, on_update);

    // Then subscribe for future changes.
    let tx_cb = Arc::clone(tx);
    let on_update_cb = Arc::clone(on_update);
    let session_cb = session.clone();
    let _ = session.MediaPropertiesChanged(&TypedEventHandler::new(
        move |_s: &Option<GlobalSystemMediaTransportControlsSession>,
              _e: &Option<windows::Media::Control::MediaPropertiesChangedEventArgs>| {
            let session = session_cb.clone();
            let tx = Arc::clone(&tx_cb);
            let on_update = Arc::clone(&on_update_cb);
            std::thread::spawn(move || {
                emit_props(&session, &tx, &on_update);
            });
            Ok(())
        },
    ));
}

fn emit_props(
    session: &GlobalSystemMediaTransportControlsSession,
    tx: &Arc<Sender<TrackInfo>>,
    on_update: &UpdateFn,
) {
    if let Ok(props) = session.TryGetMediaPropertiesAsync().and_then(|op| op.get()) {
        let title = props.Title().map(|s| s.to_string()).unwrap_or_default();
        let artist = props.Artist().map(|s| s.to_string()).unwrap_or_default();
        let album = props.AlbumTitle().map(|s| s.to_string()).unwrap_or_default();
        let cover_art = read_thumbnail(&props);
        let _ = tx.send(TrackInfo { title, artist, album, cover_art });
        on_update();
    }
}

fn read_thumbnail(
    props: &GlobalSystemMediaTransportControlsSessionMediaProperties,
) -> Option<Vec<u8>> {
    let thumb_ref = props.Thumbnail().ok()?;
    let stream = thumb_ref.OpenReadAsync().ok()?.get().ok()?;
    let size = stream.Size().ok()? as u32;
    if size == 0 {
        return None;
    }
    let reader = DataReader::CreateDataReader(&stream).ok()?;
    reader.LoadAsync(size).ok()?.get().ok()?;
    let mut bytes = vec![0u8; size as usize];
    reader.ReadBytes(&mut bytes).ok()?;
    Some(bytes)
}
