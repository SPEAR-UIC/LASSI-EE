# Security Policy

LASSI-EE is research software. It compiles and executes LLM-generated code on your machine and sends source code to the LLM backend you configure, so run it only in an environment where that is acceptable (ideally a dedicated or sandboxed GPU node).

## Reporting a vulnerability

Please report security issues privately rather than in a public issue. Use GitHub's **Report a vulnerability** option on the repository's Security tab, or email mdear2@uic.edu. Reports are handled on a best-effort basis.

## Credentials

LASSI-EE reads API keys from the `OPENAI_API_KEY` environment variable or from your own local configuration. Do not commit keys, and clear notebook outputs before sharing a notebook. If you find a credential in this repository, please report it so it can be revoked.
