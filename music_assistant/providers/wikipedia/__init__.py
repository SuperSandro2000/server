"""Wikipedia Metadata provider for Music Assistant.

Adds artist biographies sourced from Wikipedia, preferring the article matching
the user's preferred language. Resolves articles by MusicBrainz artist id:

1. Inspect the artist's MusicBrainz URL relations for a per-language Wikipedia
   link (this is the cheap path; MB's response is shared with the MusicBrainz
   metadata provider and cached).
2. If no relation matches the desired language, resolve the artist's Wikidata
   Q-id (also from the MB relations) and ask Wikidata for sitelinks, which is
   the canonical hub of cross-language Wikipedia articles.
3. Fetch the lead-paragraph summary from the Wikipedia REST API.
"""

from __future__ import annotations

from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, unquote, urlparse

import aiohttp.client_exceptions
from music_assistant_models.enums import ProviderFeature
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import MediaItemMetadata

from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.throttle_retry import Throttler
from music_assistant.models.metadata_provider import MetadataProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
    from music_assistant_models.media_items import Artist
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType
    from music_assistant.providers.musicbrainz import MusicbrainzProvider, MusicBrainzRelation


SUPPORTED_FEATURES: set[ProviderFeature] = {ProviderFeature.ARTIST_METADATA}

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return WikipediaMetadataProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    # ruff: noqa: ARG001
    return ()


class WikipediaMetadataProvider(MetadataProvider):
    """Wikipedia Metadata provider."""

    throttler: Throttler

    @property
    def priority(self) -> int:
        """Priority for this provider (lower = more preferred)."""
        # below TheAudioDB (20) so a TheAudioDB description wins when both providers have one
        return 25

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self.throttler = Throttler(rate_limit=1, period=1)

    async def get_artist_metadata(self, artist: Artist) -> MediaItemMetadata | None:
        """Fetch an artist's Wikipedia summary in the user's preferred language."""
        if not artist.mbid:
            return None

        preferred_lang = self.mass.metadata.preferred_language
        # try the user's language first, then English as a universal fallback
        languages: list[str] = [preferred_lang]
        if preferred_lang != "en":
            languages.append("en")

        relations = await self._musicbrainz_relations(artist.mbid)
        if not relations:
            return None

        titles_by_lang = _wiki_titles_by_lang(relations)

        # if any of our wanted languages aren't already covered by MB,
        # fall through to Wikidata sitelinks to fill the gaps
        missing = [lang for lang in languages if lang not in titles_by_lang]
        if missing and (qid := _wikidata_qid(relations)):
            sitelinks = await self._wikidata_sitelinks(qid, tuple(sorted(missing)))
            for lang, title in sitelinks.items():
                titles_by_lang.setdefault(lang, title)

        for lang in languages:
            if title := titles_by_lang.get(lang):
                if extract := await self._fetch_summary(lang, title):
                    return MediaItemMetadata(description=extract)
        return None

    async def _musicbrainz_relations(self, mbid: str) -> list[MusicBrainzRelation] | None:
        """Return the MusicBrainz URL relations for an artist (or None if unavailable)."""
        mb_provider = cast("MusicbrainzProvider | None", self.mass.get_provider("musicbrainz"))
        if mb_provider is None:
            return None
        try:
            details = await mb_provider.get_artist_details(mbid)
        except InvalidDataError:
            return None
        return details.relations

    @use_cache(86400 * 90, persistent=True)
    async def _wikidata_sitelinks(self, qid: str, languages: tuple[str, ...]) -> dict[str, str]:
        """Return ``{lang: article_title}`` for the requested languages on a Wikidata entity."""
        sitefilter = "|".join(f"{lang}wiki" for lang in languages)
        data = await self._get_json(
            WIKIDATA_API_URL,
            params={
                "action": "wbgetentities",
                "ids": qid,
                "props": "sitelinks",
                "sitefilter": sitefilter,
                "format": "json",
            },
        )
        if not data:
            return {}
        sitelinks = data.get("entities", {}).get(qid, {}).get("sitelinks") or {}
        result: dict[str, str] = {}
        for site_key, payload in sitelinks.items():
            if not site_key.endswith("wiki"):
                continue
            lang = site_key[: -len("wiki")]
            title = payload.get("title")
            if isinstance(title, str) and title:
                result[lang] = title
        return result

    @use_cache(86400 * 90, persistent=True)
    async def _fetch_summary(self, lang: str, title: str) -> str | None:
        """Return the lead-paragraph summary for a Wikipedia article, or None."""
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
        data = await self._get_json(url)
        if not data:
            return None
        # disambiguation pages still return 200; skip them, they aren't bios
        if data.get("type") == "disambiguation":
            return None
        extract = data.get("extract")
        if isinstance(extract, str) and extract.strip():
            return extract
        return None

    async def _get_json(
        self, url: str, params: dict[str, str] | None = None
    ) -> dict[str, Any] | None:
        """HTTP GET with throttling and a Wikipedia-compliant User-Agent."""
        headers = {
            "User-Agent": f"Music Assistant/{self.mass.version} (https://music-assistant.io)"
        }
        async with self.throttler:
            try:
                async with self.mass.http_session.get(
                    url, params=params, headers=headers
                ) as response:
                    if response.status >= 400:
                        return None
                    try:
                        return cast("dict[str, Any]", await response.json())
                    except (aiohttp.client_exceptions.ContentTypeError, JSONDecodeError):
                        return None
            except (
                aiohttp.client_exceptions.ClientConnectorError,
                aiohttp.client_exceptions.ServerDisconnectedError,
                TimeoutError,
            ):
                return None


def _wiki_titles_by_lang(relations: list[MusicBrainzRelation]) -> dict[str, str]:
    """Extract ``{lang: article_title}`` from MusicBrainz wikipedia URL relations."""
    result: dict[str, str] = {}
    for relation in relations:
        if relation.type != "wikipedia" or not relation.url:
            continue
        parsed = urlparse(relation.url.resource)
        host = parsed.netloc.lower()
        if not host.endswith(".wikipedia.org"):
            continue
        lang = host.split(".", 1)[0]
        if not parsed.path.startswith("/wiki/"):
            continue
        title = unquote(parsed.path[len("/wiki/") :])
        if title:
            result.setdefault(lang, title)
    return result


def _wikidata_qid(relations: list[MusicBrainzRelation]) -> str | None:
    """Extract the Wikidata Q-id from MusicBrainz URL relations."""
    for relation in relations:
        if relation.type != "wikidata" or not relation.url:
            continue
        candidate = relation.url.resource.rstrip("/").rsplit("/", 1)[-1]
        if candidate.startswith("Q") and candidate[1:].isdigit():
            return candidate
    return None
