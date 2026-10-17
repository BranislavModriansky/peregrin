import os
import gzip
import json
import shutil
import hashlib
import threading
from warnings import warn
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *


class HTTPStatusError(IOError):
    """Raised when the server answered with an HTTP error status (e.g. 404)."""

    def __init__(self, url, status):
        super().__init__(f"HTTP {status} for {url}")
        self.url = url
        self.status = status


def is_remote(path) -> bool:
    """True if `path` is an http(s) URL."""
    return isinstance(path, str) and path.strip().lower().startswith(('http://', 'https://'))


def _http_response(url: str, headers: dict = None, timeout: float = 60):
    """
    GET `url` and return (status, response_headers, body), requesting gzip transfer.

    Tries the stdlib first, then `requests` and `urllib3` + `certifi` (which ship
    their own CA bundle) to work around missing system certificates.
    An HTTP error status is raised immediately as HTTPStatusError, without retrying;
    304 Not Modified is returned (with an empty body), not raised.
    """
    headers = {'User-Agent': 'peregrin', 'Accept-Encoding': 'gzip', **(headers or {})}
    errors = []

    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            body = resp.read()
            if resp.headers.get('Content-Encoding', '').lower() == 'gzip':
                body = gzip.decompress(body)
            return resp.status, dict(resp.headers), body
    except HTTPError as e:
        if e.code == 304:
            return 304, dict(e.headers), b''
        raise HTTPStatusError(url, e.code) from e
    except Exception as e:
        errors.append(f"urllib: {e}")

    try:
        import requests
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            raise HTTPStatusError(url, r.status_code)
        return r.status_code, dict(r.headers), r.content
    except HTTPStatusError:
        raise
    except Exception as e:
        errors.append(f"requests: {e}")

    try:
        import urllib3, certifi
        http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())
        r = http.request("GET", url, headers=headers, timeout=float(timeout))
        if r.status >= 400:
            raise HTTPStatusError(url, r.status)
        return r.status, dict(r.headers), r.data
    except HTTPStatusError:
        raise
    except Exception as e:
        errors.append(f"urllib3: {e}")

    raise IOError(f"Could not download {url}. Attempts:\n  " + "\n  ".join(errors))


def http_get(url: str, headers: dict = None, timeout: float = 60) -> bytes:
    """Download `url` and return the raw bytes (no caching)."""
    return _http_response(url, headers, timeout)[2]


DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.peregrin', 'cache')


def _cache_dir():
    return os.environ.get('PEREGRIN_CACHE_DIR', DEFAULT_CACHE_DIR)


def clear_cache():
    """Delete all locally cached remote files."""
    shutil.rmtree(_cache_dir(), ignore_errors=True)


def ensure_cached(url: str, headers: dict = None, timeout: float = 60, immutable: bool = False) -> str:
    """
    Ensure `url` is present in the persistent on-disk cache (default
    `~/.peregrin/cache`, override with the PEREGRIN_CACHE_DIR environment
    variable) and return the local file path of the cached copy.

    The file is never held in memory here beyond the single write; callers can
    hand the returned path straight to a reader (polars, ElementTree, ...) so
    only one file is in memory at a time.

    If `immutable` is True, a cached copy is trusted forever without contacting
    the server (meant for content-addressed URLs, e.g. GitHub raw files pinned
    to a commit SHA). Otherwise, the server is asked to revalidate the cached
    copy via ETag/Last-Modified; the cache is also used, with a warning, when
    the server cannot be reached at all.
    """
    key = hashlib.sha256(url.encode()).hexdigest()
    data_path = os.path.join(_cache_dir(), key)
    meta_path = data_path + '.json'

    has_cache = os.path.isfile(data_path)
    if has_cache and immutable:
        return data_path

    conditional = dict(headers or {})
    if has_cache:
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get('etag'):
                conditional['If-None-Match'] = meta['etag']
            if meta.get('last_modified'):
                conditional['If-Modified-Since'] = meta['last_modified']
        except Exception:
            pass

    try:
        status, resp_headers, body = _http_response(url, conditional, timeout)
    except HTTPStatusError:
        raise
    except Exception as e:
        if has_cache:
            warn(f"Could not reach {url} ({e}) -> using the locally cached copy.", InputWarning)
            return data_path
        raise

    if status == 304 and has_cache:
        return data_path

    lower = {k.lower(): v for k, v in resp_headers.items()}
    os.makedirs(_cache_dir(), exist_ok=True)
    tmp = f"{data_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, 'wb') as f:
        f.write(body)
    os.replace(tmp, data_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'url': url, 'etag': lower.get('etag'), 'last_modified': lower.get('last-modified')}, f)
    return data_path


def http_get_cached(url: str, headers: dict = None, timeout: float = 60, immutable: bool = False) -> bytes:
    """Return the bytes of `url`, going through the on-disk cache (see `ensure_cached`)."""
    with open(ensure_cached(url, headers, timeout, immutable), 'rb') as f:
        return f.read()


class _LinkParser(HTMLParser):
    """Collects `href` targets of <a> tags from an HTML directory listing."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


class FileTree:

    tree_sets = {}

    ALLOWED_EXTENSIONS = ('.csv', '.xlsx', '.xls', '.xml')

    GITHUB_API = 'https://api.github.com'
    GITHUB_RAW = 'https://raw.githubusercontent.com'

    def __init__(self):
        self.tree = {}
        self.root_name = '.'

    def _is_data_file(self, name):
        return name.lower().endswith(self.ALLOWED_EXTENSIONS)

    def _build(self, root_path):
        tree = {}
        for entry in os.scandir(root_path):
            if entry.is_dir():
                tree[entry.name] = self._build(entry.path)
            elif self._is_data_file(entry.name):
                tree[entry.name] = str(os.path.join(root_path, entry.name))
        return tree

    # ---------------------------------------------------------------- remote

    @staticmethod
    def _remote_root_name(url):
        segments = [s for s in urlparse(url).path.split('/') if s]
        return unquote(segments[-1]) if segments else urlparse(url).netloc

    def _build_remote(self, url):
        github = self._parse_github_url(url)
        if github is not None:
            return self._build_github(*github)
        return self._build_http_index(url)

    # --- GitHub

    @staticmethod
    def _parse_github_url(url):
        """
        Split a github.com URL into (owner, repo, ref/path segments).
        Supports `https://github.com/<owner>/<repo>[/tree/<ref>/<path>]`.
        Returns None for non-GitHub URLs.
        """
        parsed = urlparse(url)
        if parsed.netloc.lower() not in ('github.com', 'www.github.com'):
            return None

        segments = [unquote(s) for s in parsed.path.split('/') if s]
        if len(segments) < 2:
            raise FileFinderError(f"Not a GitHub repository URL: '{url}'.")

        owner, repo = segments[0], segments[1].removesuffix('.git')
        rest = segments[2:]

        if rest and rest[0] != 'tree':
            raise FileFinderError(
                f"'{url}' does not point to a GitHub directory. "
                f"Use a URL of the form https://github.com/<owner>/<repo>/tree/<branch>/<path>."
            )

        return owner, repo, rest[1:]

    def _github_api(self, endpoint):
        headers = {'Accept': 'application/vnd.github+json'}
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
        try:
            return json.loads(http_get(f"{self.GITHUB_API}/{endpoint}", headers=headers))
        except HTTPStatusError as e:
            if e.status in (403, 429):
                raise FileFinderError(
                    "GitHub API request was refused (likely rate limit exceeded). "
                    "Set a GITHUB_TOKEN environment variable to raise the limit."
                ) from e
            raise

    def _build_github(self, owner, repo, ref_and_path):
        repo_api = f"repos/{quote(owner)}/{quote(repo)}"

        if not ref_and_path:
            ref_and_path = [self._github_api(repo_api)['default_branch']]

        # Branch names may contain '/', so try each possible ref/path split (shortest ref first).
        for i in range(1, len(ref_and_path) + 1):
            ref, path = '/'.join(ref_and_path[:i]), '/'.join(ref_and_path[i:])
            try:
                listing = self._github_api(f"{repo_api}/git/trees/{quote(ref, safe='')}?recursive=1")
            except HTTPStatusError as e:
                if e.status in (404, 422):
                    continue
                raise
            break
        else:
            raise FileFinderError(
                f"Could not resolve branch/path '{'/'.join(ref_and_path)}' in GitHub repository '{owner}/{repo}'."
            )

        # Pin file URLs to the resolved commit SHA: such URLs are immutable, so
        # downloads can be cached locally forever, while a branch update yields
        # new URLs (and thus fresh downloads) on the next make_tree call.
        try:
            raw_ref = self._github_api(f"{repo_api}/commits/{quote(ref, safe='')}")['sha']
        except Exception:
            raw_ref = ref

        if listing.get('truncated'):
            return self._build_github_contents(owner, repo, ref, path, raw_ref)

        prefix = f"{path}/" if path else ''
        if path and not any(
            item['path'] == path and item['type'] == 'tree' for item in listing['tree']
        ):
            raise FileFinderError(f"Directory '{path}' not found in GitHub repository '{owner}/{repo}' at '{ref}'.")

        tree = {}
        for item in listing['tree']:
            if not item['path'].startswith(prefix):
                continue
            parts = item['path'][len(prefix):].split('/')
            is_dir = item['type'] == 'tree'
            if not is_dir and (item['type'] != 'blob' or not self._is_data_file(parts[-1])):
                continue

            node = tree
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            if is_dir:
                node.setdefault(parts[-1], {})
            else:
                node[parts[-1]] = self._github_raw_url(owner, repo, raw_ref, item['path'])

        return self._sort_tree(tree)

    def _build_github_contents(self, owner, repo, ref, path, raw_ref=None):
        """Fallback for very large repositories: walk the Contents API folder by folder."""
        raw_ref = raw_ref or ref
        endpoint = f"repos/{quote(owner)}/{quote(repo)}/contents/{quote(path)}?ref={quote(ref, safe='')}"
        try:
            entries = self._github_api(endpoint)
        except HTTPStatusError as e:
            if e.status == 404:
                raise FileFinderError(
                    f"Directory '{path}' not found in GitHub repository '{owner}/{repo}' at '{ref}'."
                ) from e
            raise
        if not isinstance(entries, list):
            raise FileFinderError(f"'{path}' in GitHub repository '{owner}/{repo}' is not a directory.")

        tree = {}
        for entry in sorted(entries, key=lambda e: e['name']):
            if entry['type'] == 'dir':
                tree[entry['name']] = self._build_github_contents(owner, repo, ref, entry['path'], raw_ref)
            elif entry['type'] == 'file' and self._is_data_file(entry['name']):
                tree[entry['name']] = self._github_raw_url(owner, repo, raw_ref, entry['path'])
        return tree

    def _github_raw_url(self, owner, repo, ref, path):
        return f"{self.GITHUB_RAW}/{quote(owner)}/{quote(repo)}/{quote(ref)}/{quote(path)}"

    def _sort_tree(self, tree):
        return {
            name: self._sort_tree(sub) if isinstance(sub, dict) else sub
            for name, sub in sorted(tree.items())
        }

    # --- generic HTTP directory index (e.g. Apache / nginx autoindex, `python -m http.server`)

    def _build_http_index(self, url, _visited=None):
        if not url.endswith('/'):
            url += '/'
        _visited = _visited if _visited is not None else set()
        if url in _visited:
            return {}
        _visited.add(url)

        try:
            html = http_get(url).decode('utf-8', errors='replace')
        except HTTPStatusError as e:
            raise FileFinderError(f"Could not list remote directory '{url}' (HTTP {e.status}).") from e

        parser = _LinkParser()
        parser.feed(html)

        tree = {}
        for href in parser.links:
            target = urljoin(url, href)
            parsed = urlparse(target)
            if parsed.query or parsed.fragment or not target.startswith(url) or target == url:
                continue

            relative = target[len(url):]
            name = unquote(relative.rstrip('/'))
            if not name or '/' in name:
                continue  # only direct children

            if relative.endswith('/'):
                tree[name] = self._build_http_index(target, _visited)
            elif self._is_data_file(name):
                tree[name] = target

        return tree

    def _guard(self, tree=None, path='.'):
        """
        Each folder can only contain either subfolders or data files,
        but not both. Raises ValueError on the first violation.
        """
        if tree is None:
            tree = self.tree

        folders = {name: sub for name, sub in tree.items() if isinstance(sub, dict)}
        files = {name for name, sub in tree.items() if not isinstance(sub, dict)}

        if folders and files:
            raise FileFinderError(
                f"Invalid structure at '{path}': folder contains both "
                f"subfolders and data files.\n"
                f"  Subfolders: {sorted(folders)}\n"
                f"  Data files: {sorted(files)}"
            )

        for name, subtree in folders.items():
            self._guard(subtree, path=f"{path}/{name}")


    def make_tree(self, root_path):
        """
        Builds the file tree starting from the given root path (main directory).

        Parameters:
            root_path (str | PathLike): A local directory, or a remote directory URL:
                - a GitHub folder, e.g. https://github.com/<owner>/<repo>/tree/<branch>/<path>
                  (set GITHUB_TOKEN to avoid API rate limits), or
                - any http(s) URL serving an HTML directory index.
                For remote directories, the tree leaves are file URLs.

        Returns:
            FileTree: An instance of the FileTree with the constructed tree.

        Result methods:
        ----
        >>> result.show()  # Displays the tree structure
        >>> result.get('dict')  # Returns the tree as a dictionary
        """

        if is_remote(root_path):
            root_path = root_path.strip()
            self.root_name = self._remote_root_name(root_path)
            self.tree = self._build_remote(root_path)
        else:
            self.root_name = os.path.basename(os.path.abspath(root_path))
            self.tree = self._build(root_path)
        self._guard()
        return self
    

    def show(self, tree=None, root_name=None, prefix=''):
        """
        Displays the file tree structure in a readable scheme format.

        Parameters:
            tree (dict, optional): The tree structure to display. Defaults to None (uses self.tree).
            root_name (str, optional): The name of the root directory. Defaults to None (uses self.root_name).
            prefix (str, optional): The prefix for formatting the output. Defaults to ''.
        """

        if tree is None:
            tree = self.tree

        if prefix == '':
            print(root_name if root_name is not None else self.root_name)
        
        if not isinstance(tree, str):
            entries = list(tree.items())
        else:
            entries = []

        for index, (name, subtree) in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = '└── ' if is_last else '├── '
            print(prefix + connector + name)
            if subtree is not None:
                extension = '    ' if is_last else '│   '
                self.show(subtree, root_name=name, prefix=prefix + extension)

    def get(self, type: str = 'dict'):
        """
        Returns the file tree in the specified format ('dict' or 'list') 
        for further processing - loading data from the given directory.

        Parameters:
            type (str): The desired format of the tree. 
                        'dict' returns a nested dictionary structure.
                        'list' returns a list representation of the tree.
        """
        current = self.tree
        if type == 'dict':
            return current
        elif type == 'list':
            return self._tree_to_list(current)
        else:
            raise ValueError("Invalid type. Use 'dict' or 'list'.")
        

    def _tree_to_list(self, tree):
        result = []
        for name, subtree in tree.items():
            if isinstance(subtree, dict):
                result.append(self._tree_to_list(subtree))
            else:
                result.append(subtree)
        return result


make_tree = FileTree().make_tree
