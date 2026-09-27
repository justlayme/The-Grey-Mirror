#!/usr/bin/env bash
# Download the public datasets used by the case studies into data/raw/ (idempotent).
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }

fetch() {  # url file
  [ -s "$2" ] && return 0
  echo "downloading $2"
  curl -fsSL --retry 4 --retry-delay 2 -o "$2.part" "$1" && mv "$2.part" "$2"
}

# Conversations Gone Awry (Cornell ConvoKit): Wikipedia talk pages and Reddit r/ChangeMyView
fetch https://zissou.infosci.cornell.edu/convokit/datasets/conversations-gone-awry-corpus/conversations-gone-awry-corpus.zip cga-wiki.zip
fetch https://zissou.infosci.cornell.edu/convokit/datasets/conversations-gone-awry-cmv-corpus/conversations-gone-awry-cmv-corpus.zip cga-cmv.zip
# Switchboard word alignments and transcripts (Mississippi State ISIP, MS98 release)
fetch https://isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz swb_word_alignments.tar.gz

[ -f conversations-gone-awry-corpus/utterances.jsonl ] || unzip -q -o cga-wiki.zip \
  'conversations-gone-awry-corpus/conversations.json' 'conversations-gone-awry-corpus/utterances.jsonl'
[ -f conversations-gone-awry-cmv-corpus/utterances.jsonl ] || unzip -q -o cga-cmv.zip \
  'conversations-gone-awry-cmv-corpus/conversations.json' 'conversations-gone-awry-cmv-corpus/utterances.jsonl'
[ -d swb_ms98_transcriptions ] || tar xzf swb_word_alignments.tar.gz
echo "data ready in $(pwd)"
