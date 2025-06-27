# generative-grammar-prompt-nodes

**Repository Name:** Generative Grammar Prompt Nodes for InvokeAI

**Author:** dwringer

**License:** MIT

**Requirements:**
- invokeai>=4

## Introduction
![lookups usage example graph](https://raw.githubusercontent.com/dwringer/generative-grammar-prompt-nodes/main/lookuptables_usage.jpg)

This generates prompts from simple user-defined grammar rules (loaded
from custom files - examples provided below). The prompts are made by
recursively expanding a special template string, replacing nonterminal
"parts-of-speech" until no more nonterminal terms remain in the
string.

As a basic example, the template `{picture} of {subject}` might be
expanded to `{photoQuality} photo of {person}`, which might
subsequently expand to `professional photo of Barack Obama` (assuming
these definitions were present in the lookup table):

```
prompt:
  templates:
    - "{picture} of {subject}"
  picture:
    - sketch
    - cartoon
    - "{photoQuality} photo"
  photoQuality:
    - professional
    - amateur
  subject:
    - "{person}"
  person:
    - Barack Obama
    - George Bush
    - Tom Hanks
    - Rami Malek
```

### Installation:

To install these nodes, simply place the folder containing this
repository's code (or just clone the repository yourself) into your
`invokeai/nodes` folder.

Generally, the two methods of installation are:

- Open a terminal with git access (`git bash` on Windows) in
your InvokeAI home directory, and `cd` to the `nodes`
subdirectory. If you installed to `C:\Users\<user
name>\invokeai\` then you will want your terminal to be open to
`C:\Users\<user name>\invokeai\nodes\`.  Then simply type:
```
git clone https://github.com/dwringer/generative-grammar-prompt-nodes.git
```

- Or, download the source code of this repository as a .zip file by
clicking the green `<> Code` button above and selecting `Download
ZIP`, then extract the folder within and place it as a subfolder
inside the `nodes` folder of your InvokeAI home directory (e.g.,
`C:\Users\<user name>\invokeai\nodes\generative-grammar-prompt-nodes-master\`)

## Overview
### Nodes
- [Crossover Prompt](#crossover-prompt) - Performs a crossover operation on two parent seed vectors to generate a new prompt.
- [Generate Evolutionary Prompts](#generate-evolutionary-prompts) - Generates a new population of prompts by performing crossover operations
- [Lookup Table from File](#lookup-table-from-file) - Loads a lookup table from a YAML file
- [Lookups Entry from Prompt](#lookups-entry-from-prompt) - Creates a lookup table of a single heading->value
- [Prompt from Lookup Table](#prompt-from-lookup-table) - Creates prompts using lookup table templates
- [Separate Prompt and Seed Vector](#separate-prompt-and-seed-vector) - Parses a JSON string representing a list of two strings,

<details>
<summary>

### Output Definitions

</summary>

- `LookupTableOutput` - Output definition with 1 fields
- `HalvedPromptOutput` - Output definition with 5 fields
- `EvolutionaryPromptListOutput` - Output definition with 2 fields
- `JsonListStringsOutput` - Output definition with 2 fields
</details>

## Nodes
### Crossover Prompt
**ID:** `crossover_prompt`

**Category:** prompt

**Tags:** prompt, lookups, grammar, crossover, GA

**Version:** 1.4.1

**Description:** Performs a crossover operation on two parent seed vectors to generate a new prompt.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `lookups` | `Union[(str, list[str])]` | Lookup table(s) containing template(s) (JSON) | [] |
| `remove_negatives` | `bool` | Whether to strip out text between [] | False |
| `strip_parens_probability` | `float` | Probability of removing attention group weightings | 0.0 |
| `resolutions` | `Union[(str, list[str])]` | JSON structure of substitutions by id by tag | [] |
| `resolutions_dict` | `dict` | Private field for id substitutions dict cache | {} |
| `parent_a_seed_vector_in` | `str` | JSON array of seeds for Parent A's generation | None |
| `parent_b_seed_vector_in` | `str` | JSON array of seeds for Parent B's generation | None |
| `child_a_or_b` | `bool` | True for Child A (Parent A + B's branch), False for Child B (Parent B + A's branch) | True |
| `crossover_non_terminal` | `Optional[str]` | Optional: The non-terminal (key in lookups) to target for the crossover branch. If None, a random one will be chosen from available non-terminals. | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `self._core_generate_prompt(...)`



</details>

---
### Generate Evolutionary Prompts
**ID:** `generate_evolutionary_prompts`

**Category:** prompt

**Tags:** prompt, lookups, grammar, evolution, GA, population

**Version:** 1.5.0

**Description:** Generates a new population of prompts by performing crossover operations

on an input list of parent seed vectors.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `lookups` | `Union[(str, list[str])]` | Lookup table(s) containing template(s) (JSON) | [] |
| `remove_negatives` | `bool` | Whether to strip out text between [] | False |
| `strip_parens_probability` | `float` | Probability of removing attention group weightings | 0.0 |
| `resolutions` | `Union[(str, list[str])]` | JSON structure of substitutions by id by tag | [] |
| `resolutions_dict` | `dict` | Private field for id substitutions dict cache | {} |
| `seed_vectors_in` | `list[str]` | List of JSON array strings, each representing a parent's seed vector. | [] |
| `target_population_size` | `int` | The desired size of the new population to generate. | 10 |
| `selected_pair` | `int` | The selected population member to output specifically. | 0 |
| `ga_seed` | `int` | Seed for the random number generator to ensure deterministic GA operations | 0 |
| `crossover_non_terminal` | `Optional[str]` | Optional: The non-terminal (key in lookups) to target for the crossover branch. If None, a random one will be chosen from available common non-terminals for each crossover. | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `EvolutionaryPromptListOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `prompt_seed_pairs` | `list[str]` | List of JSON strings, each containing [prompt, seed_vector_json] |
| `selected_pair` | `str` | The index-selected prompt from the output population |


</details>

---
### Lookup Table from File
**ID:** `lookup_table_from_file`

**Category:** prompt

**Tags:** prompt, lookups, grammar, file

**Version:** 1.2.0

**Description:** Loads a lookup table from a YAML file

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `file_path` | `str` | Path to lookup table YAML file | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LookupTableOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lookups` | `str` | The output lookup table |


</details>

---
### Lookups Entry from Prompt
**ID:** `lookup_from_prompt`

**Category:** prompt

**Tags:** prompt, lookups, grammar

**Version:** 1.3.2

**Description:** Creates a lookup table of a single heading->value

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `heading` | `str` | Heading for the lookup table entry | None |
| `lookup` | `str` | The entry to place under Heading in the lookup table |  |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LookupTableOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lookups` | `str` | The output lookup table |


</details>

---
### Prompt from Lookup Table
**ID:** `prompt_from_lookup_table`

**Category:** prompt

**Tags:** prompt, lookups, grammar

**Version:** 1.5.0

**Description:** Creates prompts using lookup table templates

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `lookups` | `Union[(str, list[str])]` | Lookup table(s) containing template(s) (JSON) | [] |
| `remove_negatives` | `bool` | Whether to strip out text between [] | False |
| `strip_parens_probability` | `float` | Probability of removing attention group weightings | 0.0 |
| `resolutions` | `Union[(str, list[str])]` | JSON structure of substitutions by id by tag | [] |
| `resolutions_dict` | `dict` | Private field for id substitutions dict cache | {} |
| `seed_vector_in` | `Optional[str]` | Optional JSON array of seeds for deterministic generation | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `self._core_generate_prompt(...)`



</details>

---
### Separate Prompt and Seed Vector
**ID:** `separate_prompt_and_seed_vector`

**Category:** prompt

**Tags:** json, list, split, string, prompt, genetic

**Version:** 1.5.0

**Description:** Parses a JSON string representing a list of two strings,

outputting each string separately.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `pair_input` | `str` | JSON string of a list containing exactly two strings, e.g., '["string one", "string two"]' | ["", ""] |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `JsonListStringsOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `prompt` | `str` | The prompt string from the JSON list |
| `seed_vector` | `str` | The seed vector string from the JSON list |


</details>

---

## Footnotes
There are some example .yaml files here to give some ideas about what
sort of things are possible. Most of these possibilities are
illustrated by the file `example_template.yaml`.

