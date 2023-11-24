![lookups usage example graph](https://raw.githubusercontent.com/dwringer/generative-grammar-prompt-nodes/main/lookuptables_usage.jpg)

### Installation:

To install these nodes, simply place the folder containing this repository's code (or just checkout the repository yourself) into your `invokeai/nodes` folder.

### Generative Grammar-Based Prompt Nodes

This generates prompts from simple user-defined grammar rules (loaded from custom files - examples provided below). The prompts are made by recursively expanding a special template string, replacing nonterminal "parts-of-speech" until no more nonterminal terms remain in the string.

As a basic example, the template `{picture} of {subject}` might be expanded to `{photoQuality} photo of {person}`, which might subsequently expand to `professional photo of Barack Obama` (assuming these definitions were present in the lookup table):
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

**Three nodes are included:**
- *Lookup Table from File* - loads a YAML file "prompt" section (or of a whole folder of YAML's) into a JSON-ified dictionary (Lookups output)
- *Lookups Entry from Prompt* - places a single entry in a new Lookups output under the specified heading
- *Prompt from Lookup Table* - uses a Collection of Lookups as grammar rules from which to randomly generate prompts.

There are some example .yaml files here to give some ideas about what sort of things are possible. Most of these possibilities are illustrated by the file `example_template.yaml`.
