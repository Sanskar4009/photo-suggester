import hashlib
import io
import re
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import requests
import streamlit as st
from insightface.app import FaceAnalysis

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = APP_DIR / "credentials.json"
DRIVE_TOKEN_FILE = APP_DIR / "google_drive_token.json"
PHOTOS_TOKEN_FILE = APP_DIR / "google_photos_token.json"

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
PHOTOS_SCOPE = "https://www.googleapis.com/auth/photospicker.mediaitems.readonly"

DRIVE_API = "https://www.googleapis.com/drive/v3"
PHOTOS_PICKER_API = "https://photospicker.googleapis.com/v1"

MODEL_OPTIONS = ["buffalo_l", "buffalo_s"]
IMAGE_TYPES = {"jpg", "jpeg", "png", "bmp", "webp"}
IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif", "image/tiff"}
THRESHOLD = 0.40

st.set_page_config(page_title="Face Photo Album", page_icon="🖼️", layout="wide", initial_sidebar_state="expanded")


# -----------------------------------------------------------------------------
# GOOGLE OAUTH
# -----------------------------------------------------------------------------

def get_google_credentials(provider: str) -> Credentials:
    """Get OAuth credentials for Google Drive or Google Photos Picker."""
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CREDENTIALS_FILE.name}. Download an OAuth Desktop App JSON "
            "file from Google Cloud Console and put it beside this Python file."
        )

    if provider == "drive":
        scopes = [DRIVE_SCOPE]
        token_file = DRIVE_TOKEN_FILE
    elif provider == "photos":
        scopes = [PHOTOS_SCOPE]
        token_file = PHOTOS_TOKEN_FILE
    else:
        raise ValueError("Provider must be 'drive' or 'photos'.")

    credentials = None
    if token_file.exists():
        try:
            credentials = Credentials.from_authorized_user_file(str(token_file), scopes)
        except Exception:
            credentials = None

    # Existing token may lack this provider's scope. Re-authorize in that case.
    if credentials and not credentials.has_scopes(scopes):
        credentials = None

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), scopes=scopes)
        credentials = flow.run_local_server(port=0, access_type="offline", prompt="consent")

    token_file.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def save_refreshed_credentials(credentials: Credentials, provider: str) -> None:
    token_file = DRIVE_TOKEN_FILE if provider == "drive" else PHOTOS_TOKEN_FILE
    token_file.write_text(credentials.to_json(), encoding="utf-8")


def get_drive_service():
    credentials = get_google_credentials("drive")
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def bearer_headers(credentials: Credentials) -> dict:
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("Google credentials are invalid. Please connect the account again.")
    return {"Authorization": f"Bearer {credentials.token}"}


# -----------------------------------------------------------------------------
# GOOGLE DRIVE: LOAD IMAGES FROM A SELECTED FOLDER AND SUBFOLDERS
# -----------------------------------------------------------------------------

def extract_drive_folder_id(folder_url_or_id: str) -> str:
    """Accept either a Drive folder ID or a URL copied from the browser."""
    value = folder_url_or_id.strip()
    if not value:
        raise ValueError("Enter a Google Drive folder URL or ID.")

    match = re.search(r"/folders/([A-Za-z0-9_-]+)", value)
    if not match:
        match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    return match.group(1) if match else value


def list_drive_images(service, root_folder_id: str) -> list[dict]:
    """Recursively list supported images in a Google Drive folder."""
    images = []
    folders_to_visit = [(root_folder_id, "")]
    visited_folders = set()

    while folders_to_visit:
        folder_id, relative_parent = folders_to_visit.pop()
        if folder_id in visited_folders:
            continue
        visited_folders.add(folder_id)

        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    pageSize=1000,
                    pageToken=page_token,
                    fields="nextPageToken, files(id, name, mimeType)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )

            for item in response.get("files", []):
                item_id = item["id"]
                item_name = item["name"]
                mime_type = item.get("mimeType", "")
                relative_name = f"{relative_parent}/{item_name}".strip("/")

                if mime_type == "application/vnd.google-apps.folder":
                    folders_to_visit.append((item_id, relative_name))
                    continue

                extension = Path(item_name).suffix.lower().lstrip(".")
                is_supported_image = extension in IMAGE_TYPES or mime_type in IMAGE_MIME_TYPES
                if is_supported_image:
                    images.append({
                        "source": "drive",
                        "id": item_id,
                        "name": relative_name,
                    })

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return sorted(images, key=lambda item: item["name"].casefold())


def download_drive_image(service, file_id: str) -> bytes:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# GOOGLE PHOTOS PICKER: USER SELECTS PHOTOS IN GOOGLE PHOTOS
# -----------------------------------------------------------------------------

def create_photos_picker_session(credentials: Credentials) -> dict:
    response = requests.post(
        f"{PHOTOS_PICKER_API}/sessions",
        headers={**bearer_headers(credentials), "Content-Type": "application/json"},
        json={},
        timeout=30,
    )
    response.raise_for_status()
    save_refreshed_credentials(credentials, "photos")
    return response.json()


def get_photos_picker_session(credentials: Credentials, session_id: str) -> dict:
    response = requests.get(
        f"{PHOTOS_PICKER_API}/sessions/{session_id}",
        headers=bearer_headers(credentials),
        timeout=30,
    )
    response.raise_for_status()
    save_refreshed_credentials(credentials, "photos")
    return response.json()


def list_picked_photos(credentials: Credentials, session_id: str) -> list[dict]:
    """Return photo items selected by the user in a Google Photos Picker session."""
    results = []
    page_token = None

    while True:
        params = {"sessionId": session_id, "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            f"{PHOTOS_PICKER_API}/mediaItems",
            headers=bearer_headers(credentials),
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        for item in payload.get("mediaItems", []):
            media_file = item.get("mediaFile", {})
            mime_type = media_file.get("mimeType", "")
            item_type = item.get("type", "")

            # Face grouping is for still images; videos are deliberately skipped.
            if item_type == "VIDEO" or not mime_type.startswith("image/"):
                continue

            base_url = media_file.get("baseUrl")
            if not base_url:
                continue

            results.append({
                "source": "photos",
                "base_url": base_url,
                "name": media_file.get("filename") or f"google_photos_{item.get('id', uuid4().hex)}.jpg",
                "media_id": item.get("id", ""),
            })

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return sorted(results, key=lambda item: item["name"].casefold())


def download_photos_image(credentials: Credentials, base_url: str) -> bytes:
    # Picker base URLs need a download suffix and an OAuth bearer header.
    response = requests.get(
        f"{base_url}=d",
        headers=bearer_headers(credentials),
        timeout=120,
    )
    response.raise_for_status()
    save_refreshed_credentials(credentials, "photos")
    return response.content


# -----------------------------------------------------------------------------
# INSIGHTFACE
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_name):
    model = FaceAnalysis(name=model_name)
    model.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
    return model


def get_faces(image_bytes, model):
    """Return each detected face's normalized embedding and cropped preview."""
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read this image")

    faces = model.get(image)
    if not faces:
        raise ValueError("No face detected")

    height, width = image.shape[:2]
    detected_faces = []
    faces = sorted(faces, key=lambda face: float(face.bbox[0]))

    for face_number, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = np.rint(face.bbox).astype(int)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        ok, encoded_crop = cv2.imencode(".jpg", image[y1:y2, x1:x2])
        if not ok:
            continue

        detected_faces.append({
            "embedding": face.normed_embedding.astype(np.float32),
            "face_crop": encoded_crop.tobytes(),
            "face_number": face_number,
        })

    if not detected_faces:
        raise ValueError("Faces were detected, but no valid face crops could be created")
    return detected_faces


def generate_unique_key(database):
    while True:
        key = uuid4().hex[:12]
        if key not in database:
            return key


def clear_album():
    st.session_state.database = {}
    st.session_state.identity_names = {}
    st.session_state.processed_hashes = set()
    for key in list(st.session_state.keys()):
        if key.startswith("person_name_"):
            del st.session_state[key]


def get_image_bytes(source_file, drive_service=None, photos_credentials=None):
    source = source_file["source"]
    if source == "local":
        return source_file["file"].getvalue()
    if source == "drive":
        return download_drive_image(drive_service, source_file["id"])
    if source == "photos":
        return download_photos_image(photos_credentials, source_file["base_url"])
    raise ValueError(f"Unsupported image source: {source}")


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------

DEFAULTS = {
    "database": {},
    "identity_names": {},
    "processed_hashes": set(),
    "model_choice": "buffalo_l",
    "drive_connected": False,
    "photos_connected": False,
    "drive_files": [],
    "drive_folder_label": "",
    "photos_files": [],
    "photos_session_id": "",
    "photos_picker_uri": "",
    "photos_selection_loaded": False,
    "selected_identity_id": "",
    "active_view": "Library",
    "editing_identity_id": "",
    "search_people": "",
}
for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value.copy() if isinstance(default_value, (dict, list, set)) else default_value


def go_to_view(view: str) -> None:
    st.session_state.active_view = view


def open_identity(identity_id: str) -> None:
    st.session_state.selected_identity_id = identity_id
    st.session_state.active_view = "People"
    st.session_state.editing_identity_id = ""


def clear_album():
    st.session_state.database = {}
    st.session_state.identity_names = {}
    st.session_state.processed_hashes = set()
    st.session_state.selected_identity_id = ""
    st.session_state.editing_identity_id = ""
    st.session_state.search_people = ""
    for key in list(st.session_state.keys()):
        if key.startswith("person_name_") or key.startswith("name_editor_"):
            del st.session_state[key]


def unique_photos(appearances):
    """Deduplicate a person's appearances to one card per original photo."""
    by_photo = {}
    for appearance in appearances:
        key = appearance.get("image_hash") or appearance.get("name")
        by_photo.setdefault(key, appearance)
    return list(by_photo.values())


def all_indexed_photos(database):
    photos = {}
    for appearances in database.values():
        for appearance in appearances:
            key = appearance.get("image_hash") or appearance.get("name")
            photos.setdefault(key, appearance)
    return list(photos.values())


def identity_label(identity_id: str, index: int | None = None) -> str:
    name = st.session_state.identity_names.get(identity_id, "").strip()
    if name:
        return name
    if index is not None:
        return f"Person {index + 1}"
    return f"Identity {identity_id[:6]}"


# Google Photos-inspired dark styling, kept inline in this Python file (no config file needed).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&display=swap');
:root {
  --album-blue: #8ab4f8;
  --album-ink: #f1f3f4;
  --album-muted: #a7b0bb;
  --album-line: #30363d;
  --background-color: #0b0d10;
  --secondary-background-color: #15191f;
  --primary-color: #8ab4f8;
}
html, body, [class*="css"] { font-family: 'Google Sans', Arial, sans-serif; }
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: #0b0d10 !important;
  color: #f1f3f4 !important;
}
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6, .stApp [data-testid="stMarkdownContainer"] {
  color: #f1f3f4;
}
[data-testid="stHeader"] { background: rgba(11,13,16,.96) !important; }
.main .block-container { max-width: 1500px; padding-top: 1.15rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] {
  background: #111418 !important;
  border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] small, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p, .section-help, .album-subtitle,
.small-muted { color: #a7b0bb !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f3f4 !important; }
.album-brand { display:flex; align-items:center; gap:13px; margin:0 0 1.25rem 0; }
.album-logo { width:44px;height:44px;border-radius:15px;background:#1b3558;color:#8ab4f8;display:flex;align-items:center;justify-content:center;font-size:24px; }
.album-title { font-size:1.65rem;font-weight:500;letter-spacing:-.5px;color:#f1f3f4;line-height:1.2; }
.album-subtitle { font-size:.92rem;margin-top:4px; }
.section-title { font-size:1.45rem;font-weight:500;color:#f1f3f4;letter-spacing:-.3px;margin:.25rem 0 .2rem 0; }
.section-help { font-size:.94rem;margin-bottom:1rem; }
div[data-testid="stMetric"] { background:#15191f !important;border:1px solid #30363d;border-radius:18px;padding:15px 18px; }
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] label { color:#a7b0bb !important; }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div { color:#f1f3f4 !important; }
.stButton > button, [data-testid="stDownloadButton"] button {
  border-radius:999px;border:1px solid #3a424c;background:#1b2027;color:#d7e3f8;
  font-weight:500;padding:.45rem 1rem;transition:all .15s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color:#8ab4f8;background:#202b3a;color:#ffffff;
}
.stButton > button[kind="primary"] {
  background:#8ab4f8;color:#102033;border-color:#8ab4f8;
}
.stButton > button[kind="primary"]:hover { background:#a8c7fa;border-color:#a8c7fa;color:#102033; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea, .stTextInput input, textarea {
  background:#15191f !important;color:#f1f3f4 !important;
  border:1px solid #3a424c !important;border-radius:12px !important;
  caret-color:#8ab4f8 !important;
}
input::placeholder, textarea::placeholder { color:#8b949e !important; opacity:1 !important; }
[data-baseweb="select"] > div, [data-baseweb="select"] input {
  background:#15191f !important;color:#f1f3f4 !important;border-color:#3a424c !important;
}
[data-baseweb="popover"] > div, [role="listbox"], [role="option"] {
  background:#1b2027 !important;color:#f1f3f4 !important;
}
.stRadio [role="radiogroup"] { gap:8px; }
.stRadio [role="radiogroup"] label {
  background:#15191f !important;border:1px solid #30363d;border-radius:999px;
  padding:7px 14px;color:#d7dce2 !important;
}
.stRadio [role="radiogroup"] label:has(input:checked) {
  background:#263950 !important;border-color:#8ab4f8;color:#d8e8ff !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#30363d !important;border-radius:18px !important;background:#15191f !important;
}
[data-testid="stFileUploader"] section {
  background:#15191f !important;border:1px solid #30363d !important;color:#f1f3f4 !important;
}
[data-testid="stFileUploader"] section * { color:#d7dce2 !important; }
[data-testid="stImage"] img { border-radius:12px; }
[data-testid="stAlert"] { background:#15191f !important;color:#f1f3f4 !important; }
hr { border-color:#30363d !important; }
.stProgress > div > div > div > div { background:#8ab4f8 !important; }
@media (max-width:700px) {
  .album-title {font-size:1.35rem;}
  .main .block-container {padding-left:1rem;padding-right:1rem;}
}
</style>
""", unsafe_allow_html=True)

# Sidebar is deliberately simple, as in a photo-library application.
with st.sidebar:
    st.markdown("### 🖼️ Face Photo Album")
    st.caption("Your photos, organized by faces")
    st.divider()
    has_data = bool(st.session_state.database or st.session_state.processed_hashes)
    model_name = st.selectbox(
        "Face recognition model",
        options=MODEL_OPTIONS,
        key="model_choice",
        disabled=has_data,
        help="Choose before indexing. Clear the album to change models.",
    )
    st.caption(f"Similarity threshold: {THRESHOLD:.2f}")
    st.caption("Names are optional. Every detected face gets an identity.")
    st.divider()
    if st.button("Clear indexed album", use_container_width=True):
        clear_album()
        st.rerun()
    st.caption("Cloud photos are read-only. The current face index is kept in this Streamlit session.")

# Header and navigation
st.markdown(
    '<div class="album-brand"><div class="album-logo">▧</div><div><div class="album-title">Face Photo Album</div><div class="album-subtitle">A simpler way to find people in your photos</div></div></div>',
    unsafe_allow_html=True,
)
active_view = st.radio(
    "Main navigation",
    ["Library", "People", "Search", "Add photos"],
    horizontal=True,
    key="active_view",
    label_visibility="collapsed",
)

identity_count = len(st.session_state.database)
photo_count = len(all_indexed_photos(st.session_state.database))
appearance_count = sum(len(items) for items in st.session_state.database.values())

# Shared summary strip
m1, m2, m3 = st.columns(3)
m1.metric("Photos indexed", f"{photo_count:,}")
m2.metric("Unique face identities", f"{identity_count:,}")
m3.metric("Face appearances", f"{appearance_count:,}")
st.write("")

# -----------------------------------------------------------------------------
# LIBRARY VIEW
# -----------------------------------------------------------------------------
if active_view == "Library":
    st.markdown('<div class="section-title">Your library</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-help">Browse the original photos in which at least one face was detected.</div>', unsafe_allow_html=True)
    photos = all_indexed_photos(st.session_state.database)
    if not photos:
        st.info("Your library is empty. Add a local folder, Google Drive folder, or selected Google Photos to get started.")
        st.button("Add photos", type="primary", on_click=go_to_view, args=("Add photos",))
    else:
        sort_choice = st.selectbox("Sort photos", ["Filename A–Z", "Filename Z–A"], label_visibility="collapsed", key="library_sort")
        photos = sorted(photos, key=lambda item: item.get("name", "").casefold(), reverse=(sort_choice == "Filename Z–A"))
        cols_per_row = 4
        for start in range(0, len(photos), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for col, photo in zip(cols, photos[start:start + cols_per_row]):
                with col:
                    with st.container(border=True):
                        st.image(photo["image_bytes"], use_container_width=True)
                        st.caption(photo.get("name", "Photo"))

# -----------------------------------------------------------------------------
# PEOPLE VIEW: unique face identities first; names are optional.
# -----------------------------------------------------------------------------
elif active_view == "People":
    st.markdown('<div class="section-title">People</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-help">Each face is grouped automatically. Select a face to see every indexed photo associated with it. You can add a name later.</div>', unsafe_allow_html=True)

    database = st.session_state.database
    if not database:
        st.info("No identities yet. Add photos and click **Categorize photos**.")
        st.button("Add photos", type="primary", on_click=go_to_view, args=("Add photos",))
    else:
        people_search = st.text_input("Find a person", placeholder="Search by name or identity ID", key="people_search_input")
        ordered_ids = sorted(database.keys(), key=lambda ident: st.session_state.identity_names.get(ident, "").casefold() or ident)
        if people_search.strip():
            term = people_search.strip().casefold()
            ordered_ids = [ident for ident in ordered_ids if term in st.session_state.identity_names.get(ident, "").casefold() or term in ident.casefold()]

        selected_id = st.session_state.selected_identity_id
        if selected_id and selected_id in database:
            appearances = database[selected_id]
            person_photos = unique_photos(appearances)
            saved_name = st.session_state.identity_names.get(selected_id, "")
            st.divider()
            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"### {saved_name if saved_name else 'Unnamed person'}")
                st.caption(f"Identity ID: {selected_id} · {len(person_photos)} photo(s) · {len(appearances)} face appearance(s)")
            with top_right:
                st.button("← All people", on_click=open_identity, args=("",), key="back_to_people")
            face_col, description_col = st.columns([1, 4])
            with face_col:
                st.image(appearances[0]["face_crop"], width=170, caption="Face preview")
            with description_col:
                st.write("This identity contains all the photos currently grouped with this face.")
                if st.session_state.editing_identity_id == selected_id:
                    editor_key = f"name_editor_{selected_id}"
                    if editor_key not in st.session_state:
                        st.session_state[editor_key] = saved_name
                    st.text_input("Name this face (optional)", key=editor_key, placeholder="e.g. Mom, Alex, or leave blank")
                    save_col, cancel_col = st.columns([1, 1])
                    with save_col:
                        if st.button("Save name", type="primary", key=f"save_name_{selected_id}"):
                            new_name = st.session_state.get(editor_key, "").strip()
                            if new_name:
                                st.session_state.identity_names[selected_id] = new_name
                            else:
                                st.session_state.identity_names.pop(selected_id, None)
                            st.session_state.editing_identity_id = ""
                            st.rerun()
                    with cancel_col:
                        st.button("Cancel", key=f"cancel_name_{selected_id}", on_click=lambda: st.session_state.update({"editing_identity_id": ""}))
                else:
                    button_label = "Edit name" if saved_name else "＋ Add name (optional)"
                    if st.button(button_label, key=f"edit_identity_name_{selected_id}"):
                        st.session_state.editing_identity_id = selected_id
                        st.session_state[f"name_editor_{selected_id}"] = saved_name
                        st.rerun()
                    if saved_name and st.button("Remove name", key=f"remove_identity_name_{selected_id}"):
                        st.session_state.identity_names.pop(selected_id, None)
                        st.session_state.editing_identity_id = ""
                        st.rerun()

            st.divider()
            st.markdown("### Photos with this person")
            st.caption(f"Showing {len(person_photos)} unique original photo(s). A group photo can appear under multiple people.")
            for start in range(0, len(person_photos), 4):
                cols = st.columns(4, gap="medium")
                for col, photo in zip(cols, person_photos[start:start + 4]):
                    with col:
                        with st.container(border=True):
                            st.image(photo["image_bytes"], use_container_width=True)
                            st.caption(photo.get("name", "Photo"))

            st.divider()
            st.markdown("### All detected people")

        if not ordered_ids:
            st.info("No identities match that search.")
        else:
            grid_cols = 5
            for start in range(0, len(ordered_ids), grid_cols):
                cols = st.columns(grid_cols, gap="medium")
                for col, ident in zip(cols, ordered_ids[start:start + grid_cols]):
                    appearances = database[ident]
                    photos_for_person = unique_photos(appearances)
                    person_name = st.session_state.identity_names.get(ident, "")
                    with col:
                        with st.container(border=True):
                            st.image(appearances[0]["face_crop"], use_container_width=True)
                            st.markdown(f"**{person_name if person_name else 'Unnamed person'}**")
                            st.caption(f"{len(photos_for_person)} photos · ID {ident[:6]}")
                            st.button("Open person", key=f"open_identity_{ident}", use_container_width=True, on_click=open_identity, args=(ident,))

# -----------------------------------------------------------------------------
# SEARCH VIEW
# -----------------------------------------------------------------------------
elif active_view == "Search":
    st.markdown('<div class="section-title">Search your people</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-help">Names are optional. You can search using a saved name or part of an identity ID.</div>', unsafe_allow_html=True)
    search_query = st.text_input("Search", placeholder="Type a name or identity ID", key="global_person_search")
    if not st.session_state.database:
        st.info("Add and categorize photos first.")
    elif not search_query.strip():
        st.info("Enter a name or identity ID to find a face group.")
    else:
        term = search_query.strip().casefold()
        results = [ident for ident in st.session_state.database if term in st.session_state.identity_names.get(ident, "").casefold() or term in ident.casefold()]
        if not results:
            st.warning("No matching people found.")
        else:
            st.caption(f"{len(results)} matching identity/identities")
            cols_per_row = 5
            for start in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row, gap="medium")
                for col, ident in zip(cols, results[start:start + cols_per_row]):
                    appearances = st.session_state.database[ident]
                    person_name = st.session_state.identity_names.get(ident, "")
                    with col:
                        with st.container(border=True):
                            st.image(appearances[0]["face_crop"], use_container_width=True)
                            st.markdown(f"**{person_name if person_name else 'Unnamed person'}**")
                            st.caption(f"{len(unique_photos(appearances))} photos · ID {ident[:6]}")
                            st.button("View photos", key=f"search_open_{ident}", use_container_width=True, on_click=open_identity, args=(ident,))

# -----------------------------------------------------------------------------
# ADD PHOTOS VIEW: import, choose source, and index photos.
# -----------------------------------------------------------------------------
elif active_view == "Add photos":
    st.markdown('<div class="section-title">Add photos</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-help">Choose a source, load images, then build or extend your People collection.</div>', unsafe_allow_html=True)
    photo_source = st.radio(
        "Photo source",
        ["Local folder", "Google Drive folder", "Google Photos"],
        horizontal=True,
        key="photo_source_choice",
    )
    active_input_files = []

    if photo_source == "Local folder":
        local_uploaded_files = st.file_uploader(
            "Choose a folder of photos",
            type=sorted(IMAGE_TYPES),
            accept_multiple_files="directory",
            key="uploaded_folder",
            help="Supported images in the selected folder and subfolders will be included.",
        ) or []
        active_input_files = [
            {"source": "local", "name": item.name, "file": item}
            for item in local_uploaded_files
            if Path(item.name).suffix.lower().lstrip(".") in IMAGE_TYPES
        ]
        if active_input_files:
            st.success(f"Found {len(active_input_files)} image file(s) in the selected local folder.")

    elif photo_source == "Google Drive folder":
        st.write("Connect your Google account, then paste a Drive folder URL or ID. Nested subfolders are included.")
        connect_col, status_col = st.columns([1, 2])
        with connect_col:
            if st.button("Connect Google Drive"):
                try:
                    get_google_credentials("drive")
                    st.session_state.drive_connected = True
                    st.success("Google Drive connected.")
                except Exception as error:
                    st.error(f"Google Drive connection failed: {error}")
        with status_col:
            if st.session_state.drive_connected:
                st.success("Google Drive is connected.")
        if st.session_state.drive_connected:
            drive_folder_input = st.text_input(
                "Google Drive folder URL or ID",
                placeholder="https://drive.google.com/drive/folders/...",
                key="drive_folder_input",
            )
            if st.button("Load images from this Drive folder", type="primary"):
                try:
                    folder_id = extract_drive_folder_id(drive_folder_input)
                    service = get_drive_service()
                    with st.spinner("Scanning folder and subfolders..."):
                        drive_images = list_drive_images(service, folder_id)
                    st.session_state.drive_files = drive_images
                    st.session_state.drive_folder_label = drive_folder_input.strip()
                    st.success(f"Found {len(drive_images)} image(s) in Drive.")
                except Exception as error:
                    st.error(f"Could not read the Google Drive folder: {error}")
        if st.session_state.drive_files:
            st.caption(f"Loaded {len(st.session_state.drive_files)} Drive image reference(s).")
            with st.expander("Preview Drive file names"):
                for item in st.session_state.drive_files[:300]:
                    st.write(item["name"])
                if len(st.session_state.drive_files) > 300:
                    st.caption("Showing first 300 file names.")
            active_input_files = st.session_state.drive_files

    elif photo_source == "Google Photos":
        st.write("Connect your account and select photos using Google Photos Picker. The API requires explicit selection; it does not automatically expose the entire library.")
        if st.button("Connect Google Photos"):
            try:
                get_google_credentials("photos")
                st.session_state.photos_connected = True
                st.success("Google Photos connected.")
            except Exception as error:
                st.error(f"Google Photos connection failed: {error}")
        if st.session_state.photos_connected:
            if st.button("Create Google Photos selection", type="primary"):
                try:
                    credentials = get_google_credentials("photos")
                    session_data = create_photos_picker_session(credentials)
                    st.session_state.photos_session_id = session_data["id"]
                    st.session_state.photos_picker_uri = session_data.get("pickerUri", "")
                    st.session_state.photos_selection_loaded = False
                    st.session_state.photos_files = []
                    st.success("Selection created. Open the picker and choose your photos.")
                except Exception as error:
                    st.error(f"Could not create Google Photos selection: {error}")
            if st.session_state.photos_picker_uri:
                st.link_button("Open Google Photos Picker", st.session_state.photos_picker_uri)
                st.caption("After selecting photos in the new tab, return here and load the selection.")
                if st.button("Load selected Google Photos"):
                    try:
                        credentials = get_google_credentials("photos")
                        session_data = get_photos_picker_session(credentials, st.session_state.photos_session_id)
                        if not session_data.get("mediaItemsSet", False):
                            st.info("Finish selecting photos in Google Photos first.")
                        else:
                            with st.spinner("Loading selected photo references..."):
                                picked = list_picked_photos(credentials, st.session_state.photos_session_id)
                            st.session_state.photos_files = picked
                            st.session_state.photos_selection_loaded = True
                            st.success(f"Loaded {len(picked)} selected photo(s). Videos are skipped.")
                    except Exception as error:
                        st.error(f"Could not load selected Google Photos: {error}")
        if st.session_state.photos_selection_loaded and st.session_state.photos_files:
            st.caption(f"Selected images: {len(st.session_state.photos_files)}")
            st.warning("Google Photos download URLs expire. Categorize this selection soon.")
            with st.expander("Preview selected photo names"):
                for item in st.session_state.photos_files[:300]:
                    st.write(item["name"])
                if len(st.session_state.photos_files) > 300:
                    st.caption("Showing first 300 names.")
            active_input_files = st.session_state.photos_files

    st.divider()
    can_process = bool(active_input_files)
    st.markdown("### Build your People collection")
    st.caption("No naming is required. The app creates an identity for each detected face automatically.")
    if st.button("Categorize photos", type="primary", disabled=not can_process, use_container_width=True):
        processed_images = 0
        faces_processed = 0
        duplicate_count = 0
        errors = []
        total_files = len(active_input_files)
        try:
            model = load_model(model_name)
            drive_service = get_drive_service() if photo_source == "Google Drive folder" else None
            photos_credentials = get_google_credentials("photos") if photo_source == "Google Photos" else None
            progress = st.progress(0)
            with st.spinner("Finding faces and updating People..."):
                for index, source_file in enumerate(active_input_files):
                    relative_name = source_file["name"]
                    try:
                        image_bytes = get_image_bytes(source_file, drive_service=drive_service, photos_credentials=photos_credentials)
                        image_hash = hashlib.sha256(image_bytes).hexdigest()
                        if image_hash in st.session_state.processed_hashes:
                            duplicate_count += 1
                            progress.progress((index + 1) / total_files)
                            continue
                        try:
                            detected_faces = get_faces(image_bytes, model)
                        except ValueError as error:
                            st.session_state.processed_hashes.add(image_hash)
                            errors.append(f"{relative_name}: {error}")
                            progress.progress((index + 1) / total_files)
                            continue
                        st.session_state.processed_hashes.add(image_hash)
                        processed_images += 1
                        faces_processed += len(detected_faces)
                        for detected_face in detected_faces:
                            embedding = detected_face["embedding"]
                            matches = []
                            for identity_id, known_appearances in st.session_state.database.items():
                                similarity = max(float(np.dot(appearance["embedding"], embedding)) for appearance in known_appearances)
                                if similarity >= THRESHOLD:
                                    matches.append((similarity, identity_id))
                            if matches:
                                _, identity_id = max(matches, key=lambda item: item[0])
                            else:
                                identity_id = generate_unique_key(st.session_state.database)
                                st.session_state.database[identity_id] = []
                            st.session_state.database[identity_id].append({
                                "name": relative_name,
                                "image_hash": image_hash,
                                "image_bytes": image_bytes,
                                "embedding": embedding,
                                "face_crop": detected_face["face_crop"],
                                "face_number": detected_face["face_number"],
                            })
                    except Exception as error:
                        errors.append(f"{relative_name}: processing failed ({error})")
                    progress.progress((index + 1) / total_files)
            if processed_images:
                st.success(f"Processed {processed_images} photo(s) and detected {faces_processed} face(s). Open People to browse the results.")
            if duplicate_count:
                st.info(f"Skipped {duplicate_count} photo(s) already processed in this session.")
            if errors:
                with st.expander(f"Skipped or failed photos ({len(errors)})"):
                    for error in errors:
                        st.write(f"- {error}")
            if photo_source == "Google Photos" and photos_credentials:
                save_refreshed_credentials(photos_credentials, "photos")
        except Exception as error:
            st.error(f"Processing could not start: {error}")

# Footer
st.divider()
st.caption("Photos and face embeddings are kept in the current Streamlit session's memory. Google OAuth tokens are stored locally in token JSON files. The app does not modify or delete cloud photos.")

