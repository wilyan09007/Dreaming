
### Regime W: later trials of held-out tasks, memory = earlier trials (workflow re-run many times)

**retail / gpt-4o** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 97.1 |
| input-keyed | 2% | 10/10 | 97.3 | 97.4 | 0.3 | 0 | 97.1 |
| input-keyed | 5% | 10/10 | 97.3 | 97.4 | 0.3 | 0 | 97.1 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 94.2 |
| state-aware | 2% | 10/10 | 93.5 | 94.2 | 0.0 | 0 | 94.2 |
| state-aware | 5% | 10/10 | 93.5 | 94.2 | 0.0 | 0 | 94.2 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 97.1 |
| entity-scoped | 2% | 10/10 | 97.2 | 97.1 | 0.0 | 0 | 97.1 |
| entity-scoped | 5% | 10/10 | 97.2 | 97.1 | 0.0 | 0 | 97.1 |
  per tool (state-aware, 2%): get_order_details: skip 95% of 2600 err 0; get_user_details: skip 100% of 987 err 0; find_user_id_by_name_zip: skip 99% of 960 err 0; get_product_details: skip 91% of 845 err 0; find_user_id_by_email: skip 80% of 530 err 0; list_all_product_types: skip 88% of 262 err 0

**retail / sonnet-3.5** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 7/10 | 78.9 | 55.5 | 0.4 | 0 | 98.1 |
| input-keyed | 2% | 10/10 | 98.6 | 98.5 | 0.4 | 0 | 98.1 |
| input-keyed | 5% | 10/10 | 98.6 | 98.5 | 0.4 | 0 | 98.1 |
| state-aware | 1% | 10/10 | 95.8 | 96.2 | 0.0 | 0 | 96.2 |
| state-aware | 2% | 10/10 | 95.8 | 96.2 | 0.0 | 0 | 96.2 |
| state-aware | 5% | 10/10 | 95.8 | 96.2 | 0.0 | 0 | 96.2 |
| entity-scoped | 1% | 10/10 | 98.6 | 98.5 | 0.1 | 0 | 98.4 |
| entity-scoped | 2% | 10/10 | 98.6 | 98.5 | 0.1 | 0 | 98.4 |
| entity-scoped | 5% | 10/10 | 98.6 | 98.5 | 0.1 | 0 | 98.4 |
  per tool (state-aware, 2%): get_order_details: skip 97% of 6159 err 0; get_user_details: skip 99% of 2120 err 0; get_product_details: skip 93% of 2111 err 0; find_user_id_by_name_zip: skip 99% of 1920 err 0; find_user_id_by_email: skip 87% of 1312 err 0; list_all_product_types: skip 91% of 216 err 0

**retail / pooled** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 9/10 | 79.1 | 72.0 | 0.4 | 0 | 98.1 |
| input-keyed | 2% | 10/10 | 98.6 | 98.5 | 0.4 | 0 | 98.1 |
| input-keyed | 5% | 10/10 | 98.6 | 98.5 | 0.4 | 0 | 98.1 |
| state-aware | 1% | 10/10 | 97.2 | 97.1 | 0.0 | 0 | 97.1 |
| state-aware | 2% | 10/10 | 97.2 | 97.1 | 0.0 | 0 | 97.1 |
| state-aware | 5% | 10/10 | 97.2 | 97.1 | 0.0 | 0 | 97.1 |
| entity-scoped | 1% | 10/10 | 98.6 | 98.5 | 0.1 | 0 | 98.4 |
| entity-scoped | 2% | 10/10 | 98.6 | 98.5 | 0.1 | 0 | 98.4 |
| entity-scoped | 5% | 10/10 | 98.6 | 98.5 | 0.1 | 0 | 98.4 |
  per tool (state-aware, 2%): get_order_details: skip 99% of 6159 err 0; get_user_details: skip 99% of 2120 err 0; get_product_details: skip 95% of 2111 err 0; find_user_id_by_name_zip: skip 99% of 1920 err 0; find_user_id_by_email: skip 87% of 1312 err 0; list_all_product_types: skip 92% of 216 err 0

**airline / gpt-4o** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 88.4 |
| input-keyed | 2% | 0/10 | - | 0.0 | - | 0 | 88.4 |
| input-keyed | 5% | 9/10 | 88.7 | 78.1 | 0.0 | 0 | 88.4 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 82.2 |
| state-aware | 2% | 0/10 | - | 0.0 | - | 0 | 82.2 |
| state-aware | 5% | 6/10 | 82.2 | 46.4 | 0.0 | 0 | 82.2 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 88.4 |
| entity-scoped | 2% | 0/10 | - | 0.0 | - | 0 | 88.4 |
| entity-scoped | 5% | 9/10 | 88.7 | 78.1 | 0.0 | 0 | 88.4 |

**airline / sonnet-3.5** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 97.0 |
| input-keyed | 2% | 10/10 | 94.0 | 96.9 | 0.0 | 0 | 97.0 |
| input-keyed | 5% | 10/10 | 94.0 | 96.9 | 0.0 | 0 | 97.0 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 94.3 |
| state-aware | 2% | 10/10 | 89.1 | 94.3 | 0.0 | 0 | 94.3 |
| state-aware | 5% | 10/10 | 89.1 | 94.3 | 0.0 | 0 | 94.3 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 97.0 |
| entity-scoped | 2% | 10/10 | 94.0 | 96.9 | 0.0 | 0 | 97.0 |
| entity-scoped | 5% | 10/10 | 94.0 | 96.9 | 0.0 | 0 | 97.0 |
  per tool (state-aware, 2%): get_reservation_details: skip 98% of 2663 err 0; search_direct_flight: skip 87% of 967 err 0; get_user_details: skip 98% of 946 err 0; search_onestop_flight: skip 80% of 378 err 0; list_all_airports: skip 100% of 47 err 0

**airline / pooled** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 97.4 |
| input-keyed | 2% | 10/10 | 94.5 | 97.4 | 0.0 | 0 | 97.4 |
| input-keyed | 5% | 10/10 | 94.5 | 97.4 | 0.0 | 0 | 97.4 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 95.7 |
| state-aware | 2% | 10/10 | 90.5 | 95.7 | 0.0 | 0 | 95.7 |
| state-aware | 5% | 10/10 | 90.5 | 95.7 | 0.0 | 0 | 95.7 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 97.4 |
| entity-scoped | 2% | 10/10 | 94.5 | 97.4 | 0.0 | 0 | 97.4 |
| entity-scoped | 5% | 10/10 | 94.5 | 97.4 | 0.0 | 0 | 97.4 |
  per tool (state-aware, 2%): get_reservation_details: skip 99% of 2663 err 0; search_direct_flight: skip 89% of 967 err 0; get_user_details: skip 98% of 946 err 0; search_onestop_flight: skip 82% of 378 err 0; list_all_airports: skip 100% of 47 err 0

### Regime X: all trials of held-out tasks, memory = other tasks only (no same-task history)

**retail / gpt-4o** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 56.4 |
| input-keyed | 2% | 6/10 | 54.1 | 32.5 | 0.3 | 0 | 56.4 |
| input-keyed | 5% | 10/10 | 53.8 | 56.5 | 0.2 | 0 | 56.4 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 50.1 |
| state-aware | 2% | 6/10 | 52.9 | 29.5 | 0.0 | 0 | 50.1 |
| state-aware | 5% | 10/10 | 48.7 | 50.0 | 0.0 | 0 | 50.1 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 56.4 |
| entity-scoped | 2% | 9/10 | 54.8 | 52.0 | 0.0 | 0 | 56.4 |
| entity-scoped | 5% | 10/10 | 53.7 | 56.4 | 0.0 | 0 | 56.4 |
  per tool (state-aware, 2%): get_order_details: skip 27% of 3209 err 0; get_user_details: skip 29% of 1169 err 0; find_user_id_by_name_zip: skip 38% of 1157 err 0; get_product_details: skip 29% of 1002 err 0; find_user_id_by_email: skip 19% of 633 err 0; list_all_product_types: skip 43% of 313 err 0

**retail / sonnet-3.5** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 2/10 | 52.8 | 11.4 | 0.5 | 0 | 56.4 |
| input-keyed | 2% | 9/10 | 53.2 | 51.6 | 0.4 | 0 | 56.4 |
| input-keyed | 5% | 10/10 | 52.8 | 56.5 | 0.4 | 0 | 56.4 |
| state-aware | 1% | 9/10 | 50.8 | 48.3 | 0.0 | 0 | 52.9 |
| state-aware | 2% | 10/10 | 49.3 | 52.8 | 0.0 | 0 | 52.9 |
| state-aware | 5% | 10/10 | 49.3 | 52.8 | 0.0 | 0 | 52.9 |
| entity-scoped | 1% | 8/10 | 54.3 | 45.4 | 0.1 | 0 | 56.4 |
| entity-scoped | 2% | 10/10 | 52.6 | 56.3 | 0.1 | 0 | 56.4 |
| entity-scoped | 5% | 10/10 | 52.6 | 56.3 | 0.1 | 0 | 56.4 |
  per tool (state-aware, 2%): get_order_details: skip 51% of 7408 err 0; get_user_details: skip 51% of 2540 err 0; get_product_details: skip 59% of 2517 err 0; find_user_id_by_name_zip: skip 57% of 2317 err 0; find_user_id_by_email: skip 42% of 1525 err 0; list_all_product_types: skip 91% of 269 err 0

**retail / pooled** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 6/10 | 50.8 | 32.3 | 0.3 | 0 | 58.3 |
| input-keyed | 2% | 10/10 | 54.2 | 58.4 | 0.3 | 0 | 58.3 |
| input-keyed | 5% | 10/10 | 54.2 | 58.4 | 0.3 | 0 | 58.3 |
| state-aware | 1% | 10/10 | 50.7 | 54.6 | 0.0 | 0 | 54.7 |
| state-aware | 2% | 10/10 | 50.7 | 54.6 | 0.0 | 0 | 54.7 |
| state-aware | 5% | 10/10 | 50.7 | 54.6 | 0.0 | 0 | 54.7 |
| entity-scoped | 1% | 10/10 | 54.0 | 58.2 | 0.0 | 0 | 58.3 |
| entity-scoped | 2% | 10/10 | 54.0 | 58.2 | 0.0 | 0 | 58.3 |
| entity-scoped | 5% | 10/10 | 54.0 | 58.2 | 0.0 | 0 | 58.3 |
  per tool (state-aware, 2%): get_order_details: skip 53% of 10617 err 0; get_user_details: skip 54% of 3709 err 0; get_product_details: skip 61% of 3519 err 0; find_user_id_by_name_zip: skip 58% of 3474 err 0; find_user_id_by_email: skip 42% of 2158 err 0; list_all_product_types: skip 81% of 582 err 0

**airline / gpt-4o** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 16.4 |
| input-keyed | 2% | 0/10 | - | 0.0 | - | 0 | 16.4 |
| input-keyed | 5% | 0/10 | - | 0.0 | - | 0 | 16.4 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 15.1 |
| state-aware | 2% | 0/10 | - | 0.0 | - | 0 | 15.1 |
| state-aware | 5% | 0/10 | - | 0.0 | - | 0 | 15.1 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 16.4 |
| entity-scoped | 2% | 0/10 | - | 0.0 | - | 0 | 16.4 |
| entity-scoped | 5% | 0/10 | - | 0.0 | - | 0 | 16.4 |

**airline / sonnet-3.5** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 17.9 |
| input-keyed | 2% | 0/10 | - | 0.0 | - | 0 | 17.9 |
| input-keyed | 5% | 5/10 | 27.9 | 7.4 | 0.6 | 0 | 17.9 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 17.1 |
| state-aware | 2% | 0/10 | - | 0.0 | - | 0 | 17.1 |
| state-aware | 5% | 4/10 | 27.1 | 5.7 | 0.0 | 0 | 17.1 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 17.8 |
| entity-scoped | 2% | 0/10 | - | 0.0 | - | 0 | 17.8 |
| entity-scoped | 5% | 5/10 | 28.2 | 7.3 | 0.0 | 0 | 17.8 |

**airline / pooled** (read calls only; 10 random task splits, pooled test counts)
| key | budget | certified (seeds) | cal. coverage % | test: reads skipped % | test: realized error % of skipped | seeds with err > budget | oracle reusable % |
|---|---|---|---|---|---|---|---|
| input-keyed | 1% | 0/10 | - | 0.0 | - | 0 | 18.2 |
| input-keyed | 2% | 0/10 | - | 0.0 | - | 0 | 18.2 |
| input-keyed | 5% | 6/10 | 27.7 | 10.2 | 0.7 | 0 | 18.2 |
| state-aware | 1% | 0/10 | - | 0.0 | - | 0 | 17.7 |
| state-aware | 2% | 0/10 | - | 0.0 | - | 0 | 17.7 |
| state-aware | 5% | 7/10 | 25.3 | 11.7 | 0.0 | 0 | 17.7 |
| entity-scoped | 1% | 0/10 | - | 0.0 | - | 0 | 18.1 |
| entity-scoped | 2% | 0/10 | - | 0.0 | - | 0 | 18.1 |
| entity-scoped | 5% | 7/10 | 26.2 | 12.1 | 0.0 | 0 | 18.1 |