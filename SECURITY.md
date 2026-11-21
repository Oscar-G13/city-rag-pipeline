# Query service access

`/query` and `/warmup` require `Authorization: Bearer <CITY_RAG_API_KEY>`.
Generate a random value of at least 32 characters in private deployment
configuration. Keep this value and the provider keys on the server; browser
clients must call an authenticated server-side proxy. Missing configuration
returns 503 and missing/incorrect credentials return 401 before provider use.

Requests are limited to 16 KiB, queries to 2,000 characters, and provider
answers to 1,024 tokens. Each worker accepts at most 10 authenticated requests
per minute and two concurrent queries. For multiple workers or public traffic,
apply an aggregate quota at the authenticated gateway and set provider spending
limits. CORS defaults to no cross-origin access; configure exact origins only.

The built-in launch binds to loopback. Use an authenticated gateway for remote
access. Health checks at `/` do not invoke any provider. Provider errors and
query contents are not returned in error messages or logged by the application.

Install the pinned `requirements.txt`. To refresh it, run
`uv pip compile requirements.in --python 3.12 --upgrade -o requirements.txt`,
then run `pip-audit -r requirements.txt` and
`python -m unittest discover -s tests -v`. Tests use stub providers and make no
paid API calls. Review and deploy these changes to the actual hosting service;
this repository audit did not deploy a running City RAG service.
