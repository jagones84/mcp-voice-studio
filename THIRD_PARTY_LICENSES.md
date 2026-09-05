# Third-Party Licenses

`mcp-voice-studio` (this repo) is MIT-licensed (see [LICENSE](./LICENSE)).
The following third-party components are NOT distributed in this repository —
they are installed or downloaded separately at runtime — but their licenses
are listed here for full transparency.

---

## 1. VoiceStudio (debpalash) — AGPL-3.0

- Repository: https://github.com/debpalash/VoiceStudio
- License: GNU Affero General Public License v3.0
- SPDX: `AGPL-3.0-or-later`
- How it is used: `mcp-voice-studio` invokes the `omnivoice-infer` binary
  from VoiceStudio's venv as a separate subprocess. No code from VoiceStudio
  is copied, modified, or statically/dynamically linked into this repo.
  VoiceStudio is installed by the user via a separate `git clone` step
  (see [README-voice-studio-dgx.md](./README-voice-studio-dgx.md) §1).
- Why this is NOT a derivative work: the Free Software Foundation has clarified
  that invoking a GPL/AGPL program via subprocess / pipes / stdin-stdout is
  "mere aggregation" and does not extend the AGPL terms to the calling program.
- AGPL section 13 ("Remote Network Interaction"): applies **only** if you
  offer a network service that exposes VoiceStudio functionality to remote
  users. For local/personal use or distribution of this MIT-licensed wrapper,
  it does not trigger.

If you distribute VoiceStudio itself (or a derivative thereof), you must do
so under AGPL-3.0 and respect its source-availability clauses. That is
your responsibility, not the responsibility of this wrapper.

---

## 2. OmniVoice (k2-fsa) — Apache License 2.0

- Repository: https://huggingface.co/k2-fsa/OmniVoice
- License: Apache License, Version 2.0
- SPDX: `Apache-2.0`
- How it is used: model weights + tokenizer are downloaded at first run
  via `huggingface_hub` into `~/.cache/huggingface/`. Not bundled with
  this repo.
- Apache 2.0 imposes attribution + a NOTICE file. Keep the
  `k2-fsa/OmniVoice` credit in any derivative work.

---

## 3. higgs-audio-v2-tokenizer (eustlb) — MIT

- Repository: https://huggingface.co/eustlb/higgs-audio-v2-tokenizer
- License: MIT
- SPDX: `MIT`
- How it is used: downloaded as a dependency of OmniVoice from Hugging Face.
  No source code is bundled with this repo.

---

## 4. Python dependencies (this repo's `pyproject.toml`)

| Package | License | SPDX |
|---|---|---|
| `mcp` (Model Context Protocol SDK) | MIT | `MIT` |
| `pydantic` | MIT | `MIT` |
| `numpy` | BSD-3-Clause | `BSD-3-Clause` |
| `scipy` | BSD-3-Clause | `BSD-3-Clause` |

For full license texts, see the corresponding PyPI pages or the
`LICENSES_bundled.txt` files inside each package.

---

## Summary

- **This repo (`mcp-voice-studio`) is MIT.** You can publish, fork, modify,
  and sublicense it under MIT terms. Credit appreciated but not legally
  required.
- **VoiceStudio is AGPL-3.0 and stays AGPL-3.0.** You must respect its
  terms **only** if you distribute VoiceStudio itself (or modified versions
  of it). Installing it locally and invoking it via subprocess from this
  MIT wrapper does not extend AGPL-3.0 to this wrapper.
- **Apache 2.0 / MIT components** impose attribution. Keep the credits
  in the README and consider including a NOTICE file in derivative
  distributions.
