use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crossbeam_channel::Sender;

use windows::Media::Control::{
    GlobalSystemMediaTransportControlsSession,
    GlobalSystemMediaTransportControlsSessionManager,
    GlobalSystemMediaTransportControlsSessionMediaProperties,
    GlobalSystemMediaTransportControlsSessionPlaybackStatus,
};
use windows::Storage::Streams::DataReader;
use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};

#[derive(Clone, Debug)]
pub struct TrackInfo {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub cover_art: Option<Vec<u8>>,
}

pub fn log_smtc(msg: &str) {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("smtc_debug.log")
    {
        let _ = writeln!(file, "[{}] {}", now, msg);
    }
}

/// Spawns a background thread that polls Windows' System Media Transport
/// Controls (SMTC) for the current track and cover art.
///
/// `should_quit` is set to `true` by the main app when it wants the thread
/// to stop. The thread runs while `should_quit` is `false`.
///
/// `on_update` is called after every `TrackInfo` that was sent, so the UI can
/// wake up from idle (the UI no longer repaints on its own).
pub fn spawn_smtc_listener<F>(
    tx: Sender<TrackInfo>,
    should_quit: Arc<AtomicBool>,
    session_count: Arc<AtomicUsize>,
    on_update: F,
) where
    F: Fn() + Send + 'static,
{
    std::thread::spawn(move || {
        log_smtc("SMTC: worker thread started.");

        // ── 1. Initialize COM on this worker thread ───────────
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        }

        // ── 2. Acquire the session manager, retrying on failure ─
        let manager = loop {
            // Check quit before each retry attempt.
            if should_quit.load(Ordering::Relaxed) {
                log_smtc("SMTC: quitting before manager acquired.");
                return;
            }

            match GlobalSystemMediaTransportControlsSessionManager::RequestAsync()
                .and_then(|op| op.get())
            {
                Ok(m) => {
                    log_smtc("SMTC: session manager acquired successfully.");
                    break m;
                }
                Err(e) => {
                    log_smtc(&format!(
                        "SMTC: failed to get session manager: {:?} (retrying)",
                        e
                    ));
                    std::thread::sleep(Duration::from_millis(1000));
                }
            }
        };

        let mut last_title = String::new();
        let mut last_artist = String::new();
        let mut last_album = String::new();

        // ── 3. Poll loop — runs while should_quit is FALSE ────
        while !should_quit.load(Ordering::Relaxed) {
            // Poll slower when there is no media session at all.
            let mut poll_ms: u64 = 1000;
            match manager.GetSessions() {
                Ok(sessions) => {
                    let count = match sessions.Size() {
                        Ok(c) => c as usize,
                        Err(e) => {
                            log_smtc(&format!("SMTC: sessions.Size() failed: {:?}", e));
                            0
                        }
                    };
                    session_count.store(count, Ordering::Relaxed);

                    let session = pick_best_session(&sessions);
                    if session.is_none() {
                        poll_ms = 2000;
                    }

                    if let Some(session) = session {
                        if let Ok(props) =
                            session.TryGetMediaPropertiesAsync().and_then(|op| op.get())
                        {
                            let title =
                                props.Title().map(|s| s.to_string()).unwrap_or_default();
                            let artist =
                                props.Artist().map(|s| s.to_string()).unwrap_or_default();
                            let album = props
                                .AlbumTitle()
                                .map(|s| s.to_string())
                                .unwrap_or_default();

                            if title != last_title
                                || artist != last_artist
                                || album != last_album
                            {
                                let cover_art = read_thumbnail(&props);
                                last_title = title.clone();
                                last_artist = artist.clone();
                                last_album = album.clone();

                                log_smtc(&format!(
                                    "SMTC: now playing → {} — {} (cover: {})",
                                    artist,
                                    title,
                                    cover_art
                                        .as_ref()
                                        .map(|b| format!("{} bytes", b.len()))
                                        .unwrap_or_else(|| "none".into())
                                ));

                                let _ = tx.send(TrackInfo {
                                    title,
                                    artist,
                                    album,
                                    cover_art,
                                });
                                on_update();
                            }
                        }
                    } else if !last_title.is_empty() {
                        // Nothing is playing anymore — clear the UI.
                        log_smtc("SMTC: no playing session found. Clearing track info.");
                        last_title.clear();
                        last_artist.clear();
                        last_album.clear();
                        let _ = tx.send(TrackInfo {
                            title: String::new(),
                            artist: String::new(),
                            album: String::new(),
                            cover_art: None,
                        });
                        on_update();
                    }
                }
                Err(e) => {
                    log_smtc(&format!("SMTC: GetSessions failed: {:?}", e));
                    session_count.store(0, Ordering::Relaxed);
                }
            }

            std::thread::sleep(Duration::from_millis(poll_ms));
        }

        log_smtc("SMTC: polling loop exited. Thread shutting down.");
    });
}

/// Prefer a session whose playback status is `Playing`; otherwise fall back
/// to the first session that has any usable properties.
fn pick_best_session(
    sessions: &windows::Foundation::Collections::IVectorView<
        GlobalSystemMediaTransportControlsSession,
    >,
) -> Option<GlobalSystemMediaTransportControlsSession> {
    let count = sessions.Size().ok()?;
    let mut fallback: Option<GlobalSystemMediaTransportControlsSession> = None;

    for idx in 0..count {
        let session = match sessions.GetAt(idx) {
            Ok(s) => s,
            Err(_) => continue,
        };

        if is_playing(&session) {
            return Some(session);
        }

        if fallback.is_none() {
            fallback = Some(session);
        }
    }

    fallback
}

fn is_playing(session: &GlobalSystemMediaTransportControlsSession) -> bool {
    session
        .GetPlaybackInfo()
        .and_then(|info| info.PlaybackStatus())
        .map(|status| status == GlobalSystemMediaTransportControlsSessionPlaybackStatus::Playing)
        .unwrap_or(false)
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
