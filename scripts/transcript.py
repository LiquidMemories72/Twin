from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import json
import yt_dlp

def get_video_metadata(url):
    with yt_dlp.YoutubeDL({
        "quiet": True
    }) as ydl:

        info = ydl.extract_info(
            url,
            download=False
        )

        return {
            "title": info.get("title"),
            "upload_date": info.get("upload_date"),
            "channel": info.get("uploader"),
            "duration": info.get("duration"),
        }

TRANSCRIPT_DIR = Path("data/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

urls = ["https://www.youtube.com/watch?v=MWMe7yjPYpE"
]

api = YouTubeTranscriptApi()


def get_video_id(url):
    parsed = urlparse(url)

    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    return parse_qs(parsed.query)["v"][0]


metadata = []

for url in urls:

    try:
        video_id = get_video_id(url)

        print(f"Fetching {video_id}")

        transcript = api.fetch(video_id)

        text = " ".join(
            snippet.text
            for snippet in transcript
        )

        txt_path = TRANSCRIPT_DIR / f"{video_id}.txt"

        with open(
            txt_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(text)

        video_meta = get_video_metadata(url)

        metadata.append(
            {
                "id": video_id,
                "title": video_meta["title"],
                "upload_date": video_meta["upload_date"],
                "year": int(video_meta["upload_date"][:4]),
                "channel": video_meta["channel"],
                "duration": video_meta["duration"],
                "source_type": "transcript",
                "language": transcript.language,
                "url": url,
                "file": str(txt_path)
            }
        )

        print(f"Saved {video_id}")

    except Exception as e:

        print(f"Failed {url}")
        print(e)

with open(
    TRANSCRIPT_DIR / "transcripts_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"\nCollected {len(metadata)} transcripts.")