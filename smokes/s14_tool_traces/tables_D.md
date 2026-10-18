
**retail**: 2018 write calls (transfer excluded), 209 error returns (base rate 10.4%); error rate per tool: cancel_pending_order 1% of 291; exchange_delivered_order_items 18% of 416; modify_pending_order_address 9% of 227; modify_pending_order_items 12% of 458; modify_pending_order_payment 28% of 18; modify_user_address 0% of 119; return_delivered_order_items 10% of 489
  strongest LR features (std. coef): tool=exchange_delivered_order_items +0.65, item_ids__len +0.60, same_tool_prior -0.57, ret.status=pending +0.51, ret.status=delivered -0.51, tool=modify_pending_order_items -0.38, user_gap -0.35, tool=cancel_pending_order -0.35

**airline**: 744 write calls (transfer excluded), 148 error returns (base rate 19.9%); error rate per tool: book_reservation 39% of 149; cancel_reservation 0% of 188; send_certificate 0% of 30; update_reservation_baggages 9% of 66; update_reservation_flights 29% of 287; update_reservation_passengers 0% of 24
  strongest LR features (std. coef): flights__len +0.50, payment_id~certificate_# +0.49, payment_id~credit_card_# -0.41, cabin~business +0.39, ret.flight_type=one_way -0.36, tool=cancel_reservation -0.31, id_seen_frac -0.30, passengers__len +0.29
| domain | model | AUC (grouped 5-fold by task) | AUC 95% CI (task bootstrap) | avg precision | base rate | precision @ top-20% | recall @ top-20% |
|---|---|---|---|---|---|---|---|
| retail | tool-only LR | 0.585 | [0.508, 0.675] | 0.133 | 10.4% | 17% | 33% |
| retail | all-context LR | 0.740 | [0.644, 0.818] | 0.334 | 10.4% | 26% | 50% |
| retail | all-context GBM | 0.795 | [0.724, 0.868] | 0.474 | 10.4% | 29% | 56% |
| retail | per-tool base rate | 0.598 | [0.517, 0.688] | 0.142 | 10.4% | 18% | 34% |
| airline | tool-only LR | 0.643 | [0.510, 0.752] | 0.252 | 19.9% | 15% | 16% |
| airline | all-context LR | 0.850 | [0.780, 0.914] | 0.549 | 19.9% | 56% | 56% |
| airline | all-context GBM | 0.789 | [0.712, 0.862] | 0.477 | 19.9% | 46% | 47% |
| airline | per-tool base rate | 0.654 | [0.516, 0.764] | 0.260 | 19.9% | 14% | 14% |