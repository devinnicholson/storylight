# Nominal absence in voice descriptions

The demo baseline revealed that “A red lobster walks without any notebooks.” could produce a positive notebooks object. The dependency compiler treated `without` as an ordinary preposition and appended the absent object to the action.

The compiler now emits a typed `additional_object` negative for a complete nominal absence attached directly to a single event. The absent noun is excluded from positive objects and action text. Verbal, nested, counted, coordinated or ambiguous scope is rejected for review rather than guessed. Named absences also require review. This is a bounded extension; it does not claim coverage of arbitrary negation.

Eighteen retained cases were parsed with spaCy 3.8.11 and its 3.8.0 model, including positive `with` neighbors and unsupported scopes. Independent review passed, as did 1,241 repository Python tests, both workbench JavaScript suites and Ruff.

The parser and API bridge revisions advance together to v5, preventing reuse of old parser proofs. Deploy both services together when integrating. This commit does not deploy to the Jetson, change the best-demo checkpoint, or alter the frozen V5 experiment validator and datasets.
