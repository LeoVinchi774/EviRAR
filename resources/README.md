# Bundled Cards

The default configuration reads cards from the two directories beside this file.

Each directory contains a fixed 500-card example subset of the original research collection. 

## Knowledge cards

Filenames follow `KC_<four-digit ID>_<entry name>.md`, for example `KC_0001_伤风.md`. IDs 0001–0500 were assigned by sorting the original filenames; they are repository identifiers, not national-standard terminology codes. The entry name comes from the card’s `证候名称` heading. Identical entry names remain separate cards with distinct IDs. Existing terminology codes and all clinical content remain inside each card unchanged. Keep assigned IDs when adding future cards rather than renumbering existing files. Renaming changes file paths, so any existing knowledge-vector cache is rebuilt automatically on the next retrieval run.

`knowledge/` contains 500 Markdown syndrome entries. Entries represent normalized syndrome names and may include standard identifiers, differentiation-system labels, clinical manifestations, tongue/pulse features, etiology, differentiation rules, differential syndromes, and structured metadata. The parser supports the heading and metadata formats used in the bundled corpus.

## Consultation-experience cards

`experience/` contains 500 Markdown entries used for semantic retrieval of consultation strategy and diagnostic reasoning examples. Each file is treated as one retrieval document. Do not assume that inclusion in this repository establishes permission to redistribute the content.

## Provenance and redistribution

The bundle was copied from a local research artifact when this standalone project was assembled. A card-by-card source manifest and redistribution permissions have not yet been confirmed. The repository's Apache-2.0 software license does not grant rights to these cards. Verify the source and rights for the national-standard-derived knowledge entries and every experience document before publishing or redistributing this directory. Remove or replace any entry containing personal or otherwise restricted information unless its use is authorized.

Embedding caches are created under the local user's cache directory and are not part of this resource bundle.
