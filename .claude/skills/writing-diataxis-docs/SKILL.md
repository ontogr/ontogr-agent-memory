---
name: writing-diataxis-docs
description: Standard for writing and reviewing user-facing documentation. Covers page types, behavior-first prose with no page narration, structure, headings, formatting by meaning, code examples, callouts, links, contracts, and review criteria. Use when writing or editing pages under docs/, the README, or other user-facing Markdown or MDX.
---

# Documentation standard

This skill governs documentation pages: what each page is for, how it is structured, how its prose behaves, and how it is formatted and reviewed. Page outlines are in [TEMPLATES.md](TEMPLATES.md).

One editorial rule sits above the others. Describe the software, its behavior, and the reader's task, and never narrate the documentation.

## Precedence

Prose style follows `writing-plain`, as `AGENTS.md` says. This skill decides
page type, structure, headings, lists, tables and modal verbs.

## What binds this repo

- User-facing docs live under `docs/`. Update them when code behavior changes.
- Link to other pages with relative file paths such as `../concepts/how-data-flows.mdx`, and run `make docs-lint-links`. Link to a section with its anchor when the reader needs one contract.
- Do not edit the generated pages in `docs/docs/api/` or the frozen copies in `docs/versioned_docs/`. Describe only current behavior.
- Do not hard-wrap prose to 88 characters. Wrap only Python code, docstrings, and comments.
- Use real commands, data formats, symbols, and paths, and verify every claim against the code before you write it.
- Docs never mention ADRs, plans, research notes, or benchmarks by name.
- Python blocks in `docs/docs/guides/` and `docs/docs/get-started/` run in `make docs-test`. Mark a block that cannot run on its own with `python notest`.

## Describe the software and the task

Every sentence defines something, explains behavior, states a condition, names a trade-off, gives an instruction, reports an expected result, or helps the reader choose. A sentence that only says where information lives has no place on the page.

| Avoid | Prefer |
|---|---|
| Each strategy has its own page with the full detail. | Use [fixed-size chunking](#) when chunks must stay within a token limit. |
| This section explains configuration. | Set `timeout_seconds` to limit how long a request can wait. |
| Below is an example of how to create a client. | Create a client with your API key. |
| The index, as covered earlier, stores embeddings. | The index stores one embedding per chunk. |
| Next we look at the different search options. | Keyword search matches terms. Vector search compares embeddings. |
| See the relevant documentation for more information. | [Configure retries](#) to handle temporary connection failures. |
| The following table lists all available options. | Delete the sentence when the heading already names the table. |
| You can easily customize the powerful pipeline. | Pass a custom extractor to `extractor`. |

When you meet a self-referential sentence, ask what the important fact is. State the fact, compare the choices and link their names, or delete the sentence. A more polished announcement is still an announcement.

Some signposting helps the reader act, so keep it. "Run the commands in order" states an execution requirement. "The example reuses the client created in Set up the connection" names a dependency.

Links point to the next task, an alternative, an explanation, or an exact contract. Name the task or the contract in the link text, and put the link where the need arises. Avoid link text that only invites a click, and avoid references to position such as "the table above". Name the thing, such as "the configuration table".

## Write about behavior

Lead with the object, the behavior, the requirement, or the result. Open a page with a definition or an outcome, and never with a description of the page or praise for the product. Use the software as the subject when you explain behavior, and use an imperative when you give an instruction. Use present tense and do not promise future features.

Keep one purpose in a paragraph. Start with the fact or the instruction, then give its condition or consequence. Split a paragraph when it moves from definition to procedure or from normal behavior to an exception. Keep articles and complete sentences, so write "The request fails if `API_KEY` is unset" and not a clipped fragment. Use one noun for one object, and define a term before you use it to explain another term.

| Purpose | Pattern | Example |
|---|---|---|
| Definition | Thing, is, category, distinguishing behavior | A retriever selects documents that match a query. |
| Behavior | Actor, verb, object | The loader preserves the document's source path. |
| Instruction | Verb, object, purpose or condition | Set `batch_size` to limit records per request. |
| Conditional instruction | If or to, condition, command | If the server requires authentication, set `API_KEY`. |
| Constraint | Thing, cannot or requires, concrete limit | The importer requires a unique ID for each record. |
| Trade-off | Choice, benefit, cost | Larger batches reduce request overhead but use more memory. |
| Result | Operation, observable outcome | The command prints the collection ID. |

## Pick the page type

Two questions pick the type. Does the content guide action or explain theory, and does it serve learning or work?

| Action or theory | Learning | Work |
|---|---|---|
| Action | Tutorial | How-to guide |
| Theory | Explanation | Reference |

A tutorial leads the learner to a result by doing, and every step produces something the learner can see. A how-to guide completes a task for a reader who already has the competence, with prerequisites, ordered actions, a check, and the failure cases that matter. A reference states facts for lookup, which are exact names, types, defaults, limits, and errors, and it mirrors the structure of the thing it describes. An explanation covers one bounded topic with its definition, mechanism, constraints, trade-offs, and reasons, and it is the only mode where opinion belongs.

Do not mix types on one page. A short example inside a concept page is fine. A complete integration tutorial inside a parameter reference is not, so link to the other type instead. Give each page one primary job and let that job set the order. A tutorial follows the learner's actions, and a reference follows the object's contract.

| Reader intent | Page | The opening establishes | The page is organized by |
|---|---|---|---|
| Assess the product | Overview | The outcome, the boundaries, the deployment choices | Capabilities and fit |
| Get a first result | Quickstart | What the reader produces and what it needs | One executable path |
| Understand a concept | Explanation | The definition and the role | Causes, relationships, trade-offs |
| Finish a known task | How-to | The target state and the conditions | Actions in dependency order |
| Choose an implementation | Catalog or comparison | The decision and the criteria | Alternatives compared on the same fields |
| Configure one implementation | Component or integration page | Purpose, required services, input and output contract | Setup, usage, configuration, limits |
| Look up a value | Reference | The exact name and scope | Stable fields, types, defaults, constraints |
| Recover from a failure | Troubleshooting | A recognizable symptom and the affected operation | Diagnostic checks and causes |
| Upgrade | Migration guide | Affected versions and breaking behavior | Old contract, replacement, verification |

## Structure the documentation

| Section | The reader asks | Include | Keep elsewhere |
|---|---|---|---|
| Overview | What does the system do, and does it fit my task? | Capabilities, boundaries, a short workflow | A full configuration catalog |
| Getting started | How do I get one result? | Requirements, install, one runnable path, the expected result | Every provider and option |
| Concepts | What are the objects and how do they relate? | Definitions, lifecycle, data flow, constraints | Long setup procedures |
| How-to guides | How do I finish this task? | Prerequisites, ordered actions, a check, relevant failures | A tour of the product |
| Components or integrations | What does this part accept, return, and require? | Purpose, inputs, outputs, install, minimal usage, compatibility | Architecture repeated on each page |
| Configuration | Which setting controls this? | Names, types, defaults, scope, precedence, examples | Unrelated tutorial steps |
| API reference | What is the exact contract? | Signatures, parameters, returns, exceptions, versions | Persuasion or feature marketing |
| Troubleshooting | Why did this fail and what fixes it? | Symptoms, causes, checks, fixes | Unordered tips |

### Index pages and detail pages

An index helps the reader choose. Give each option its mechanism, its suitable inputs, its important limit, and a descriptive link. A row that holds only a name and an invitation to read more fails. A detail page helps the reader use one option, with the mechanism, requirements, input and output contract, minimal usage, configuration, and failure cases. Link back to the comparison when the reader may reconsider.

### Usage pages and mechanics pages

Keep usage and mechanics on separate pages. A tool or component page opens with a one-sentence definition and a contract table, then an overview, then usage in each place the tool runs, then the result. A concept page explains how the tool works inside, and the tool page links to it with a sentence that says what the concept page covers. A reader who wants to call the tool never has to read the internals first, and a reader who wants the internals never has to scroll past call examples.

### Order, splits, and ownership

Order navigation by dependency. Put install and the first working result before optional integrations, and definitions before guides that rely on the terms. Keep exact API reference as its own destination and not the tail of a learning path. Use one organizing dimension within a group, so a provider group lists providers and a task group lists tasks.

Split a page when the audience, the task, the version, or an independently configurable subsystem changes. Keep closely dependent steps together, because length alone does not justify a split. Keep the sentence that states a prerequisite on the task page even when the prerequisite has its own guide, because a link does not replace saying that the requirement exists.

Give each definition and contract one owner. A guide summarizes a concept in one sentence and links the noun to its canonical page, and a quickstart does not copy the settings table. Repeat a safety constraint wherever the risky action occurs, because that repetition prevents harm. Let the decision set the page length: a family with one real choice needs a short table, and a family with several trade-offs needs explanation before its catalog. A short page stays short, and a template section with nothing useful in it is left out.

## Write headings

Use one page title. If the site builds the title from front matter, add no second H1. Use H2 for the page's main questions or tasks, H3 for subdivisions of one H2, and H4 only when a real fourth level helps navigation. Never skip a level to get smaller text, and never use a bold paragraph where a section needs an anchor.

Use simple headings that a new reader understands at once, in plain words with no jargon, no metaphors, and no invented labels. Name a usage section "Example", a list of environments "Where you can use the tools", and a ranking description "How keyword search works". Use sentence case, except for product names and symbols whose case is part of their spelling.

| Heading kind | Rule | Example |
|---|---|---|
| Task | Begins with a verb | Create the index |
| Concept | Names the subject | Document identity |
| Reference | Uses the exact name | `batch_size`, Parameters |
| Generic | Avoid. Say what the section does. | Configure authentication, or Verify the connection |

Do not repeat the page title in each heading. A reader who scans only the headings should understand the task and find each subcase. Keep a heading's anchor stable when you reword it. In MDX, set an explicit id with the comment form, such as `## Verify the connection {/* #verify-the-connection */}`.

## Choose formatting by meaning

Plain Markdown is the portable baseline. Tabs, steps, cards, callouts, and accordions depend on the framework, so keep a readable fallback for each. A pipe table is an extension to the base Markdown specification, so check the target before relying on one.

| Element | Use it for | Do not use it for |
|---|---|---|
| Paragraph | Definitions, behavior, constraints, reasoning | A sequence the reader must run in order |
| Inline code | Symbols, keys, flags, paths, literal values | General concepts or whole sentences |
| Bold | UI labels, or a short distinction the reader might miss | Every noun, bullet lead-ins, decoration |
| Bulleted list | Independent alternatives, capabilities, conditions | Ordered steps |
| Numbered list | Actions with a meaningful order | Unranked alternatives |
| Table | Comparable facts with stable columns | Long prose or multi-step instructions inside cells |
| Fenced code | Commands, runnable examples, configuration, output | A symbol that fits in the sentence |
| Note | A qualification that does not block success | Required setup or the main explanation |
| Tip | An optional improvement after the main path works | An unstated requirement |
| Warning | Data loss, cost, security, or another concrete risk | Ordinary background |
| Tabs | Mutually exclusive providers, languages, or platforms | Consecutive steps, or required information hidden in a panel |
| Cards | A few useful destinations or choices | Detail needed to finish the current task |
| Accordion | Optional diagnostic detail or long supplementary material | Prerequisites, security constraints, required commands |
| Diagram or image | Relationships, state changes, or a UI detail that words convey badly | Decoration or an unexplained screenshot |

Write list items in parallel form: all actions, all conditions, or all noun phrases with a description. Keep an ordered list only when order changes the outcome. If a step needs a code block and its result, keep both inside that step.

Choose table columns that answer a decision. A comparison needs criteria that tell the choices apart, a parameter table needs defaults and effects, and an error table needs symptoms and fixes. A single description column hides the question. Keep units in the column heading, and keep procedures and large code blocks out of cells. Move a complex row to its own subsection and leave a short linked summary.

Use one decision dimension for each tab group, and put shared prerequisites, constraints, and the intended result outside the tabs. Describe a diagram by what it shows, so alt text states the relationships or the UI state and does not repeat the file name. A diagram supplements a definition and never replaces one, and essential meaning never lives only in color, icons, or hover state. Use a blockquote for a real quotation, a horizontal rule only for a real document boundary, and inline text for requirements and caveats. Do not leave obsolete commands struck through on the main path.

### Show a worked example with real output

Give every tool or component page a worked example that a new reader can follow without reading anything else first. Use one small shared sample, with its own setup page, so that the examples on different pages build on each other. Pose a plain question, show the call, and show what it prints. Print results in a short readable form and shorten long ids, and keep a full result in a collapsed block for readers who want every field.

Run the code and paste the output that it printed. Do not remove awkward rows or tidy a surprising result, because the awkward rows teach the limits of the tool. Add output-only snippets for the cases a reader will meet: an empty result, a cut-off result, a bad input, and a changed document. If you cannot run the example, label its output as illustrative, and never describe it as real.

### Write usable code blocks

State the purpose before the code. Include imports and setup when the example is meant to run on its own, and name the objects a continuing snippet reuses. Label the language, and keep commands apart from their output. Leave shell prompts out of commands the reader will copy. Mark placeholders and say where their values come from, and never use real credentials.

Follow the example with the expected result when it is not obvious. Label illustrative output as illustrative, and do not invent exact logs, ids, timings, or model responses. Keep enough of an error message that the reader can search for it. Explain the parameter that changes the behavior, and do not narrate every line. Put a constraint immediately before the command or setting it governs.

Show standalone use before pipeline use when both exist, and separate execution contexts that really differ. Show each state change where it happens, and shorten repeated setup when only one line changes between variants. Mark cleanup separately from the success path, and label destructive cleanup.

### Use callouts for consequences

A callout interrupts the reading order, so it needs a reason. Name the consequence in the title or the first sentence. Keep required prerequisites in the main procedure. Put a warning before the action it guards. Do not stack notes to avoid organizing the page, and do not use a callout as a banner.

## Write technical detail precisely

Open with the answer. A definition names the thing and its distinguishing behavior, a procedure states the outcome, and a reference identifies the scope and the required values. Put a version or provider limitation before the example, and put a requirement beside the operation it governs. Put the contract table near the top of a component page.

Separate a default from a recommendation. "The default is 100" describes behavior, and "Use 100" is advice that needs a reason or a condition. Do not call a default optimal. Do not claim speed, quality, or ease without a measured basis, and replace praise with the capability.

Give each contract field its facts.

| Fact | What to state |
|---|---|
| Name | The exact public symbol, key, flag, or endpoint |
| Type | The accepted type and its structure |
| Required | Always, never, or under a named condition |
| Default | The effective behavior when the field is omitted |
| Units and bounds | The unit, the limits, and the invalid values |
| Scope and precedence | Where it applies, and which source wins when several set it |
| Effect | What changes when the value changes |
| Failure | The observable behavior for an invalid value |

Do not invent a value to fill a cell, and verify each fact against the implementation. Keep an unverified fact out of finished prose, and do not use "may" to hide an unknown contract.

Explain state changes. Distinguish accepting input, processing it, persisting it, indexing it, and making it searchable, and say which states a call reaches and what else is needed. State whether a retry can create a duplicate, whether the operation updates or appends, and which identity it uses. Put destructive behavior, external calls, and paid model usage before the example that triggers them.

Start an error section from a symptom the reader can recognize, such as a status code, an exception name, or a missing result. Separate possible causes from confirmed ones, and give a diagnostic before a fix when several causes fit. Use a symptom, cause, and fix table for short independent cases and a heading for a case that needs commands or branches. End with a check that confirms recovery.

Treat version status as part of the contract. Put a deprecation or experimental notice before the first affected example, and name the version, the replacement, or the missing guarantee. A migration compares the old and new behavior, and it does not interleave old and new calls in one unlabeled sequence.

## Patterns that transfer

| Situation | Do this |
|---|---|
| A page introduces a unit of work | Define the unit before you show its variants. |
| A page needs a service, version, or permission | State the requirement before the install and connection steps. |
| An operation writes data under an access boundary | Place the authorization warning at the write operation and name the exact boundary. |
| A feature collects data | Explain what it collects together with how to turn it off. |
| A component page | Put the contract table near the top, then behavior, then the smallest useful call, then pipeline or integration use, then optional configuration. |
| An operation order changes the output | Number the order, and state preservation constraints beside the setting that causes them. |
| Options look alike | Distinguish them in the overview, before the reader copies an example. |
| A catalog lists many entries | Describe each entry by what it does differently. |
| A troubleshooting page | Start with a check the reader can run, then explain what each result means. |
| An operation is short | Give it one example and no subsections. |

## Docusaurus syntax

| Feature | Syntax and rules |
|---|---|
| Admonitions | `:::note`, `:::tip`, `:::info`, `:::warning`, and `:::danger`, with an optional title in brackets such as `:::note[Avoid in-place mutation]`. Leave blank lines around the body. |
| Tabs | Import `Tabs` from `@theme/Tabs` and `TabItem` from `@theme/TabItem`. Give each `TabItem` a `value` and a `label`, use `groupId` to synchronize related choices, and use stable labels for the same choice. |
| Optional detail | `<details>` with a `<summary>`. Headings inside hidden panels can still appear in the page contents, so avoid them. |
| Comments | MDX does not accept HTML comments. Use `{/* */}`. |
| Page contents | H2 and H3 show by default. |
| Plain Markdown fallback | A labeled subsection for a tab, and "Warning:" before the action for a warning. |

### Diagram placeholders

When a diagram is planned and not drawn, add a placeholder block in the place the diagram will go. Do not reference an image file that does not exist.

```mdx
:::info[Diagram placeholder: passage-search]
Files: `docs/static/img/docs/passage-search.png` and `docs/static/img/docs/passage-search-dark.png`.

Alt text: A sentence that states the relationships the diagram shows.

Drawing prompt: The layout, the boxes, and their labels, using real function names, statuses, and relations from the code.
:::
```

Replace the block with the `Diagram` component when the images exist. A drawing that shows renamed tools gets a new file name when frozen versions of the docs still use the old image.

## Edit a page

1. Establish the facts first. Identify the public API, the current examples, the supported version, and the expected outcome. For configuration, check defaults and precedence. For an operation, check input, output, state changes, and failure behavior.
2. Write the primary reader question in working notes, and arrange the headings to answer it in order. Remove inherited template sections that do not help.
3. Write the core fact first. Add the smallest example that shows it. Then explain the condition, result, or trade-off the example cannot show.
4. Remove self-reference. For each sentence that announces a page, a table, or an example, ask for the important fact and state it, or delete the sentence.
5. Choose formatting after the prose. Do not wrap a weak paragraph in a callout, split a continuous procedure into cards, or turn every paragraph into bullets. Check the headings alone, the table headers alone, and the link text alone. Each must make sense without an introduction that announces it.
6. Verify the publication. Run `make docs-lint-links` and the docs build. Execute code examples when the environment allows, with disposable data for destructive ones. If an example cannot run, do not call its output verified. Check that copyable commands carry no prompts, that anchors still resolve after heading edits, and that required text survives a plain-text export.

## Acceptance criteria

| Area | Ready | Needs revision |
|---|---|---|
| Opening | States a fact, an outcome, or when the feature applies | Describes the page or praises the product |
| Structure | Serves one primary reader need | Mixes tasks, or hides reference facts in a tutorial |
| Contract | Names required values and observable behavior | Says "configure correctly" or leaves defaults open |
| Procedure | Has prerequisites, actions, and a success check | Skips setup or ends without a recognizable result |
| Choice | Gives criteria and trade-offs | Says "best" or "recommended" with no reason |
| Safety | States the consequence and the mitigation before the action | Hides the risk in a later paragraph or a collapsed panel |
| Examples | State their scope and supported inputs | Depend on unnamed state or invent exact output |
| Links | Name the concept, task, or contract | Say "here" or announce that more information exists |
| Formatting | Uses each element for one meaning | Uses emphasis and components to cover weak organization |
| Evidence | Matches the version and the implementation | Copies behavior from another product's docs |

## Review checklist

- Does the opening state a behavior, a definition, an outcome, or a decision?
- Can the reader tell when the feature applies and when it does not?
- Does each sentence add a fact, an action, a condition, a result, or a useful choice?
- Are page announcements and empty transitions gone?
- Do headings name distinct tasks or subjects, with no skipped levels?
- Do technical names match the code and the configuration?
- Does each link name its destination or purpose?
- Are required steps visible without opening an optional panel?
- Do tables compare like facts and hold no long instructions?
- Are code examples complete for their stated scope and free of real secrets?
- Are expected results verified, or labeled illustrative?
- Do warnings come before the actions they guard?
- Is each framework component optional, with a readable fallback?
- Are unsupported guarantees, performance claims, and vague praise removed?
- Does each fact have one owning page?
