"""Download datasets from public Nextcloud folder shares.

Usage in a notebook:

    from nextcloud_data import get_lightdark_data
    datasets = get_lightdark_data()

To add a new dataset, write another get_<name>_data() function at the bottom
that calls fetch_share() with its own hardcoded link and post-processing.
"""

import os
import fnmatch
from urllib.parse import urlparse, unquote, quote
import xml.etree.ElementTree as ET

import requests
from tqdm.auto import tqdm

_NS = {"d": "DAV:"}
_DAV_MARKER = "/public.php/webdav"
_PROPFIND_BODY = (
    '<?xml version="1.0"?>'
    '<d:propfind xmlns:d="DAV:"><d:prop>'
    "<d:resourcetype/><d:getcontentlength/>"
    "</d:prop></d:propfind>"
)


class NextcloudShare:
    """A public Nextcloud folder share, accessed via WebDAV."""

    def __init__(self, share_url, password=""):
        u = urlparse(share_url)
        token = u.path.rstrip("/").split("/")[-1]
        # handles https://host/s/TOKEN, /index.php/s/TOKEN, /nextcloud/s/TOKEN
        prefix = u.path.split("/s/")[0].replace("/index.php", "").rstrip("/")
        self.dav_url = f"{u.scheme}://{u.netloc}{prefix}{_DAV_MARKER}"
        self.session = requests.Session()
        self.session.auth = (token, password)  # public shares: user = token
        self.session.headers["X-Requested-With"] = "XMLHttpRequest"

    def _url(self, rel):
        return f"{self.dav_url}/{quote(rel)}"

    def list_files(self, rel=""):
        """Recursively list (relative_path, size) for all files in the share."""
        r = self.session.request(
            "PROPFIND", self._url(rel),
            headers={"Depth": "1"}, data=_PROPFIND_BODY,
        )
        r.raise_for_status()
        out = []
        for resp in ET.fromstring(r.content).findall("d:response", _NS):
            href = unquote(resp.find("d:href", _NS).text)
            path = href.split(_DAV_MARKER, 1)[1].strip("/")
            if path == rel.strip("/"):
                continue  # the folder itself
            if resp.find(".//d:resourcetype/d:collection", _NS) is not None:
                out += self.list_files(path)
            else:
                size = resp.find(".//d:getcontentlength", _NS)
                out.append((path, int(size.text) if size is not None and size.text else None))
        return out

    def download(self, rel, local_path, size=None):
        """Download one file; skip if it already exists with the right size."""
        if os.path.exists(local_path) and (size is None or os.path.getsize(local_path) == size):
            return
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        tmp = local_path + ".part"
        with self.session.get(self._url(rel), stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        os.replace(tmp, local_path)


def fetch_share(share_url, output, exclude=(), password="", progress=True):
    """Mirror a public Nextcloud folder share into `output`.

    Files matching any glob in `exclude` (matched against the path inside
    the share) are skipped. Already complete files are not downloaded again.
    Returns the list of downloaded paths relative to the share root.
    """
    share = NextcloudShare(share_url, password)
    files = share.list_files()
    fetched = []
    for path, size in tqdm(files, disable=not progress, desc=os.path.basename(output) or output):
        if any(fnmatch.fnmatch(path, pat) for pat in exclude):
            continue
        share.download(path, os.path.join(output, path), size)
        fetched.append(path)
    return fetched


# ---------------------------------------------------------------------------
# Dataset-specific getters. Add new get_<name>_data() functions below.
# ---------------------------------------------------------------------------

def get_lightdark_data(share_url, output="data/lightdark", exclude=("*.tif",)):
    """Download the light/dark dataset and return the sorted dataset IDs."""
    fetched = fetch_share(share_url, output, exclude=exclude)
    return sorted({
        "_".join(os.path.basename(p).split("_")[:3])
        for p in fetched
        if p.endswith(".bob2")
    })


# Example template for a future dataset:
#
# def get_otherdataset_data(output="data/otherdataset", exclude=()):
#     share_url = "https://cloud.example.org/s/XYZ"
#     fetch_share(share_url, output, exclude=exclude)
#     return ...  # whatever post-processing that dataset needs
