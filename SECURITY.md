# Security

Report suspected vulnerabilities privately through GitHub's **Report a vulnerability** option in the repository's Security tab, if available. Otherwise, use an existing private channel with the maintainer. Do not put credentials, recordings, transcripts, or personal data in a public issue.

Include the affected commit, a minimal reproduction using synthetic input, the expected behavior, and the observed impact. Redact tokens, account identifiers, device addresses, and local paths from logs. Share only the evidence needed to reproduce the problem.

Keep credentials in local configuration or a secret manager. If a credential has been exposed, revoke or rotate it; deleting a file or commit does not invalidate it. Use synthetic scenes and speech when testing privacy boundaries or preparing public examples.
