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
git clone https://github.com/dwringer/composition-nodes.git
```

- Or, download the source code of this repository as a .zip file by
clicking the green `<> Code` button above and selecting `Download
ZIP`, then extract the folder within and place it as a subfolder
inside the `nodes` folder of your InvokeAI home directory (e.g.,
`C:\Users\<user name>\invokeai\nodes\composition-nodes-master\`)

## Overview
### Nodes
- [Lookup Table from File](#lookup-table-from-file) - Loads a lookup table from a YAML file
- [Lookups Entry from Prompt](#lookups-entry-from-prompt) - Creates a lookup table of a single heading->value
- [Prompt from Lookup Table](#prompt-from-lookup-table) - Creates prompts using lookup table templates

<details>
<summary>

### Output Definitions

</summary>

- `LookupTableOutput` - Output definition with 1 fields
- `HalvedPromptOutput` - Output definition with 4 fields
</details>

## Nodes
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

**Version:** 1.3.2

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


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `HalvedPromptOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `prompt` | `str` | The output prompt |
| `part_a` | `str` | First part of the output prompt |
| `part_b` | `str` | Second part of the output prompt |
| `resolutions` | `str` | JSON dict of [tagname,id] resolutions |


</details>

---

## Footnotes
There are some example .yaml files here to give some ideas about what
sort of things are possible. Most of these possibilities are
illustrated by the file `example_template.yaml`.

