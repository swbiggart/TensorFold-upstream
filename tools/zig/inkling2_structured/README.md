# Inkling-2 structured output: masked oracle cases

One file per case, written by `tools/zig/inkling2_ref.py --mode checkpoint --structured all --grammar-lib
LIBTFGRAMMAR --xgrammar-check --cases-out DIR` (checkpoint mode, greedy, every step's pick the argmax over the
tokens the grammar allows; bf16 matmuls reduced in fp32, as `--fp32-reduction`; the prompt rendered as the server
renders it: a named `tool_choice` offers only that tool).

| Field | Meaning |
|---|---|
| `request` | the OpenAI chat request a server gate sends (temperature 0) |
| `grammar`, `grammar_kind`, `grammar_sha` | the structural tag (libtfgrammar kind 5) the case was masked with, and the first 16 hex of its sha256 |
| `stops`, `vocab` | the stop ids and the mask width given to the grammar library |
| `prompt_ids`, `prompt_text`, `prompt_sha` | the chat template's rendering of `request` (no generation prompt: the model opens its first part) |
| `ids`, `text`, `tokens`, `ids_sha` | the masked greedy reply, ending with the stop id that ends the grammar; `ids_sha` is the first 16 hex of the sha256 of the ids as little-endian int32 |
| `ended` | the grammar ended (a stop was accepted) |
| `mask_steps`, `mask_sha` | masks filled, and the first 16 hex of the sha256 of all of them in order (6,282 int32 words each) |
| `masked_picks` | steps where the mask changed the argmax |
| `mask_check` | `equal`: every mask from libtfgrammar was byte-equal to Python xgrammar 0.2.8's (compiler cache off) |
| `plain_request`, `plain_ids`, `plain_text`, `plain_tokens`, `plain_ids_sha` | the same prompt's unmasked greedy reply: `request` without `response_format` / `tool_choice` / `parallel_tool_calls` (the same prompt tokens), at most 160 tokens (`--plain-steps`), ending at a stop id |
| `first_masked_pick` | the first position where the masked reply leaves the unmasked one |
| `fp32_reduction` | the oracle ran with fp32 reductions |

A server's masked greedy reply equals the oracle when its token ids equal `ids` (the server may leave out the final
stop id). The gate: for every case whose unmasked server reply (`plain_request`, max_tokens 160) equals `plain_ids`,
every masked server reply (alone and sharing rounds) equals `ids`. A case whose unmasked reply already leaves
`plain_ids` meets an engine-vs-oracle near-tie on its prompt's own greedy path (the checkpoint oracle is not bit-equal
to the engine); its masked reply is reported, not gated.
