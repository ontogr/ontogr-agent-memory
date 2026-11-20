# Page templates

These outlines are starting points. Omit a section that has nothing useful in it, and never publish an empty heading or filler to satisfy a template. Mirror the real object. A short page stays short.

Each opening names a fact, an outcome, or a condition. It never describes the page.

## Concept

```markdown
# Document identity

<Define the object and the behavior that distinguishes it.>

## How identity is assigned

<Explain the mechanism and the lifecycle.>

## Updates and duplicates

<State what changes, what stays stable, and the limits.>

## Choose an identity policy

<Explain the trade-off with concrete criteria.>
```

Order: definition, why or when it matters, how it works, an example, limits.

## How-to guide

```markdown
# Configure a provider

<State the outcome and when the configuration is needed.>

## Prerequisites

<List only requirements that block this task.>

## Set the credentials

<Give the action and the command.>

## Select the provider

<Show the minimal configuration.>

## Verify the connection

<Give a check and its expected result.>

## Resolve authentication failures

<Give a symptom, a check, and a fix.>
```

Every prerequisite appears before its first use, and every step has a visible result.

## Component or integration reference

```markdown
# ComponentName

<State what the component does and when to use it.>

## Requirements

<List packages, credentials, services, and compatibility constraints.>

## Inputs and outputs

<Name the exact types, the required values, and the observable result.>

## Example

<Pose a plain question against the shared sample, show the call, and paste the output of a real run. Add output-only snippets for an empty result, a cut-off result, and a bad input.>

## Usage

<Give one complete minimal example. Show standalone use before pipeline use when both exist.>

## Configuration

<Give names, types, defaults, and the behavior each setting changes.>

## Limitations

<State unsupported inputs and the failure behavior.>
```

Put the contract table near the top. Add a section for a real feature only: duplicate policy, operation order, security constraints.

For a tool that runs in several places, such as Python, an agent, and a server, give Usage one subsection for each place and keep the contract table to names, inputs, outputs, requirements, and the API link. Put the internals on a separate concept page and link it from Related with a sentence that says what that page covers.

## Strategy comparison

```markdown
# Choose a chunking strategy

<Name the decision and the constraints that affect it.>

| Strategy | Mechanism | Use when | Limit |
| --- | --- | --- | --- |
| <Descriptive link> | <How it splits> | <Concrete condition> | <Concrete trade-off> |

## Preserve document structure

<Explain the choice for this requirement.>

## Control chunk size

<Explain the choice for this requirement.>
```

Each row distinguishes the options by behavior. A row that holds only a name fails.

## Troubleshooting

```markdown
# Resolve a connection timeout

<Identify the symptom and the affected operation.>

## Check the server address

<Give the diagnostic command and the expected result.>

## Check authentication

<Give the condition, the check, and the fix.>

## Adjust the timeout

<Explain when the change is appropriate and what it cannot fix.>
```

Start with the cheapest check that tells the causes apart. End with an observable check that confirms recovery.

## Quickstart

A quickstart reaches one recognizable result. It does not demonstrate every capability.

1. State what the reader creates, stores, retrieves, or runs.
2. List blocking prerequisites: versions, credentials, and external services.
3. Give one install path. Move uncommon platform variants to named alternatives.
4. Configure the minimum required values, and say where each placeholder value comes from.
5. Run the smallest complete workflow.
6. Show the result the reader can inspect. Explain nondeterminism and do not promise an exact generated response.
7. Explain only the concepts needed to read that result.
8. Give cleanup when the example leaves data or services running, and label destructive cleanup.
9. Link the next practical task by name.

An optional no-key or mock path states what it proves and which real behavior it does not exercise.

## Configuration reference

- Open with the behavior that the settings control and the scope they apply to.
- Group related settings by subsystem or task.
- Use a table with exact names, types, defaults, units, and applicability.
- Explain precedence in prose or an ordered list. Distinguish an absent value from one that is empty or disabled when the implementation treats them differently.
- Put a minimal working configuration after the settings it uses. Add an override example only when it shows a real precedence rule.
- Say which changes need a new client, a process restart, or a re-index, when verified.
- End with specific invalid combinations and a way to validate. "Check your configuration" is not a validation method.

## Integration guide

Name the product and the deployment mode on both sides, and state what data crosses the boundary. Distinguish a local library integration from a hosted account.

| Section | Content when it applies |
|---|---|
| Requirements | Supported versions, packages, accounts, services, credentials |
| Permissions | Minimum scopes, data access, authorization boundary |
| Installation | The dependency specific to the integration |
| Configuration | Endpoint, credential location, provider selection, mode precedence |
| Minimal workflow | A complete connection and one useful operation |
| Verification | A response, a stored record, or another observable result |
| Runtime behavior | Persistence, retries, sessions, sync, background work, cost |
| Troubleshooting | Symptoms, checks, and recovery specific to the integration |
| Limitations | Unsupported modes, deployment limits, version limits |

Do not praise an integration. State the setup it needs and what it does. Keep an external provider's billing, permissions, and data retention separate from this library's behavior.

## Migration guide

- Name the source and target versions in the opening. Say who is affected and which behavior changes. Keep unchanged behavior out of the steps unless the reader needs it to read a change.
- Use an old and new table for renamed symbols and settings. Use separate labeled code blocks for behavior changes.
- Order actions by dependency: preparation, data or configuration change, code change, verification, cleanup.
- State compatibility windows and rollback only when the product supports them. Name irreversible data changes before the command that makes them.
- End with a check that exercises the new observable contract.
- Link to legacy documentation only when readers still maintain an older supported version.
