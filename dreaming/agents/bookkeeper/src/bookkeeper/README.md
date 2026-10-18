# dreaming-bookkeeper

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

A bookkeeping agent. Send it one bank-feed line (the descriptor, then the amount, like `ADOBE *WEB 54.99`) and it books the
transaction: it categorizes the expense, looks up the vendor, matches an invoice, flags anything odd for a human, and posts
the entry to the ledger. The reply is a receipt: what was posted and, for each step, who answered it.

Under the hood every step runs through Dreaming: a runtime that watches an agent's tool and model calls, trains small models
on them, and lets a trained model ("a dream") answer a step only under a certificate (weights hash + error budget). The
receipt names the weights and certificate for every step a dream answered, plus the run's cost and latency.

Keywords: bookkeeping, accounting, bank feed, transaction categorization, expense, ledger, invoice matching, vendor lookup,
reconciliation, certificate, Dreaming.

Example: `NOTION *WEB 96.00`
