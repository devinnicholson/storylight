# Complete scene scope and authored display steps

The default planner selects a focal event. An experimental `scene` scope also retains supported
secondary actors, their objects and actions, spatial bindings, and explicit event order. It first
validates the model's selected action and result, then consumes the remaining source clauses with
the finite local grammar. Unknown clauses and ambiguous bindings refuse; it cannot substitute a
correct source result for an incorrect model result.

Scene scope requires all three settings:

```text
BOOKFORGE_LIVE_SCENE_PLANNER=model
BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND=tensorrt_accepted_graph
BOOKFORGE_LIVE_SCENE_PLANNER_SCOPE=scene
```

These settings have not been promoted on the appliance. They use the same accepted model request;
the additional coverage is deterministic reconstruction, not a model improvement. Semantic cache
identity and completed Story Pack lookup distinguish the two scopes. Packs record their scope.
Scene refusal cannot return a focal fallback, including when the local model connection fails.

The scene provider validates the graph and exact compiled renderer prompt before paid preparation.
It skips generic previews. Its current subject-count contract supports one to four subjects; inline
visual checks require one shared subject label. Mixed species need a richer visual check. Long
styles, oversized prompts, transformations and ordered events refuse before paid preparation.
The ordinary single-image endpoint cannot prove a temporal sequence.

`scene_playback.py` supplies local construction for two authored display steps. A transformation
replaces its source entity with the typed result, retaining the result's count and color. It does
not invent an owner or an action for that result. An ordered pair retains its first event and then
its second event without inventing postconditions such as an open basket.

Independent actions, motions, active relationships and events must have explicit phase assignments.
Static facts remain unless they refer to the consumed transformation source. Each derived graph
and renderer plan is validated against the original source passage. Display metadata contains
bounded identifiers and hashes; these bind the steps consistently but do not authenticate an
external author. The manifest makes no semantic-accuracy or visual-fidelity claim.

Projector navigation now commits the page identity, reading cursor and controls only after the new
media succeeds. A failed transition leaves the visible page active, so retry does not skip the
unseen step. The JavaScript check exercises the actual navigation functions with controlled media
loading; it is not a physical projector rehearsal.

`build_display_story_pack` assembles static and ordered steps from typed `DisplaySourcePage` inputs
into a local Story Pack and a separate hash manifest. Its authored six-page control produces eight
display pages. The pack has no generated assets and is explicitly marked as a local authored plan.
Graph-specific renderer negatives omit blanket duplicate-actor/person/tool prohibitions that can
conflict with explicit counts; the accepted plain-slot renderer remains unchanged.

Private replay of the original seven model responses under complete-scene scope produces six
proved graphs, including both foxes on the supported pages. Page 6 refuses without a focal fallback.
Evidence is in `benchmarks/product-fidelity-scene-scope-2026-09-04/`; replay makes no inference calls
or latency claim. The model-selected result remains the next isolated issue to investigate.

Generated image review and physical playback remain separate acceptance steps. No cloud image or
hardware success follows from these local checks alone.
