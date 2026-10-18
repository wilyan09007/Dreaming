tau2-bench banking_knowledge (MIT): replayed the gold actions of 97 tasks with the deterministic tools, no LLM (retrieval variant no_knowledge): 955 calls, 457 change the database or user database, 196 are read-only get_* calls. One scripted trajectory per task, so this measures the tools, not an agent.

Across tasks (chronological by task order), read calls seen in an earlier task: input-keyed 28/196 = 14.3%, of which the earlier return equals the new one 23/28; keyed on (tool, args, full database hash before the call) 6/196 = 3.1%, equal 6/6.
| read tool | calls | seen in earlier task % | earlier return equal % |
|---|---|---|---|
| get_bank_account_transactions_9173 | 53 | 7.5 | 100.0 |
| get_all_user_accounts_by_user_id_3847 | 40 | 15.0 | 100.0 |
| get_debit_cards_by_account_id_7823 | 38 | 7.9 | 100.0 |
| get_pending_replacement_orders_5765 | 18 | 16.7 | 100.0 |
| get_user_dispute_history_7291 | 17 | 41.2 | 71.4 |
| get_closure_reason_history_8293 | 10 | 0.0 | 0.0 |
| get_credit_limit_increase_history_4829 | 5 | 40.0 | 50.0 |
| get_payment_history_6183 | 5 | 40.0 | 50.0 |