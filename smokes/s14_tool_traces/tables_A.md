
**retail / gpt-4o** (460 episodes, 3274 tool calls, 4 trials; chronological = trial-major, so trial 0 is cold start)
| tool | kind | calls | (i) seen % | (i) equal % | (ii) seen % | (ii) equal % | (iii) seen % | (iii) equal % | seen only in other tasks % |
|---|---|---|---|---|---|---|---|---|---|
| calculate | pure | 56 | 35.7 | 100.0 | 35.7 | 100.0 | 35.7 | 100.0 | 5.4 |
| think | pure | 50 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| get_order_details | read | 1029 | 85.1 | 99.5 | 79.9 | 100.0 | 85.0 | 100.0 | 16.0 |
| get_user_details | read | 389 | 86.9 | 100.0 | 86.6 | 100.0 | 86.9 | 100.0 | 13.1 |
| find_user_id_by_name_zip | read | 386 | 88.1 | 100.0 | 88.1 | 100.0 | 88.1 | 100.0 | 14.0 |
| get_product_details | read | 359 | 89.4 | 100.0 | 78.6 | 100.0 | 89.4 | 100.0 | 21.2 |
| find_user_id_by_email | read | 208 | 67.3 | 100.0 | 67.3 | 100.0 | 67.3 | 100.0 | 14.4 |
| list_all_product_types | read | 106 | 99.1 | 100.0 | 80.2 | 100.0 | 99.1 | 100.0 | 48.1 |
| return_delivered_order_items | write | 160 | 69.4 | 93.7 | 69.4 | 93.7 | 69.4 | 93.7 | 8.8 |
| exchange_delivered_order_items | write | 151 | 53.0 | 96.2 | 53.0 | 96.2 | 53.0 | 96.2 | 6.6 |
| modify_pending_order_items | write | 149 | 61.7 | 77.2 | 61.7 | 77.2 | 61.7 | 77.2 | 11.4 |
| cancel_pending_order | write | 92 | 70.7 | 100.0 | 70.7 | 100.0 | 70.7 | 100.0 | 7.6 |
| modify_pending_order_address | write | 71 | 69.0 | 71.4 | 69.0 | 71.4 | 69.0 | 71.4 | 9.9 |
| modify_user_address | write | 38 | 65.8 | 100.0 | 65.8 | 100.0 | 65.8 | 100.0 | 7.9 |
| transfer_to_human_agents | write | 25 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| modify_pending_order_payment | write | 5 | 60.0 | 100.0 | 60.0 | 100.0 | 60.0 | 100.0 | 0.0 |

Read calls reusable (state-aware key) by trial index: t0: 44%, t1: 89%, t2: 94%, t3: 95%

**retail / sonnet-3.5** (920 episodes, 7086 tool calls, 8 trials; chronological = trial-major, so trial 0 is cold start)
| tool | kind | calls | (i) seen % | (i) equal % | (ii) seen % | (ii) equal % | (iii) seen % | (iii) equal % | seen only in other tasks % |
|---|---|---|---|---|---|---|---|---|---|
| think | pure | 146 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| calculate | pure | 47 | 40.4 | 100.0 | 40.4 | 100.0 | 40.4 | 100.0 | 2.1 |
| get_order_details | read | 2426 | 93.1 | 98.9 | 90.8 | 100.0 | 92.8 | 100.0 | 7.6 |
| get_product_details | read | 848 | 94.8 | 100.0 | 87.7 | 100.0 | 94.8 | 100.0 | 11.6 |
| get_user_details | read | 844 | 93.5 | 98.7 | 92.8 | 100.0 | 93.4 | 99.1 | 6.8 |
| find_user_id_by_name_zip | read | 773 | 92.0 | 100.0 | 92.0 | 100.0 | 92.0 | 100.0 | 8.2 |
| find_user_id_by_email | read | 487 | 80.3 | 100.0 | 80.3 | 100.0 | 80.3 | 100.0 | 10.5 |
| list_all_product_types | read | 84 | 98.8 | 100.0 | 89.3 | 100.0 | 98.8 | 100.0 | 26.2 |
| return_delivered_order_items | write | 329 | 85.7 | 97.5 | 85.7 | 97.5 | 85.7 | 97.5 | 4.3 |
| modify_pending_order_items | write | 309 | 77.3 | 84.5 | 77.3 | 84.5 | 77.3 | 84.5 | 4.5 |
| exchange_delivered_order_items | write | 265 | 77.4 | 97.6 | 77.4 | 97.6 | 77.4 | 97.6 | 3.4 |
| cancel_pending_order | write | 199 | 84.4 | 98.2 | 84.4 | 98.2 | 84.4 | 98.2 | 5.0 |
| modify_pending_order_address | write | 156 | 85.3 | 82.7 | 85.3 | 82.7 | 85.3 | 82.7 | 3.2 |
| modify_user_address | write | 81 | 81.5 | 100.0 | 81.5 | 100.0 | 81.5 | 100.0 | 3.7 |
| transfer_to_human_agents | write | 79 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| modify_pending_order_payment | write | 13 | 69.2 | 100.0 | 69.2 | 100.0 | 69.2 | 100.0 | 0.0 |

Read calls reusable (state-aware key) by trial index: t0: 48%, t1: 93%, t2: 94%, t3: 96%, t4: 97%, t5: 97%, t6: 96%, t7: 97%

**retail / pooled** (1380 episodes, 10360 tool calls, 8 trials; chronological = trial-major, so trial 0 is cold start)
| tool | kind | calls | (i) seen % | (i) equal % | (ii) seen % | (ii) equal % | (iii) seen % | (iii) equal % | seen only in other tasks % |
|---|---|---|---|---|---|---|---|---|---|
| think | pure | 196 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| calculate | pure | 103 | 44.7 | 100.0 | 44.7 | 100.0 | 44.7 | 100.0 | 4.9 |
| get_order_details | read | 3455 | 95.1 | 99.1 | 92.3 | 100.0 | 94.8 | 100.0 | 5.8 |
| get_user_details | read | 1233 | 95.7 | 99.2 | 95.1 | 100.0 | 95.6 | 99.4 | 4.9 |
| get_product_details | read | 1207 | 96.4 | 100.0 | 89.5 | 100.0 | 96.4 | 100.0 | 8.2 |
| find_user_id_by_name_zip | read | 1159 | 94.7 | 100.0 | 94.7 | 100.0 | 94.7 | 100.0 | 5.7 |
| find_user_id_by_email | read | 695 | 80.3 | 100.0 | 80.3 | 100.0 | 80.3 | 100.0 | 9.6 |
| list_all_product_types | read | 190 | 99.5 | 100.0 | 85.8 | 100.0 | 99.5 | 100.0 | 31.1 |
| return_delivered_order_items | write | 489 | 88.5 | 95.6 | 88.5 | 95.6 | 88.5 | 95.6 | 3.9 |
| modify_pending_order_items | write | 458 | 79.9 | 82.5 | 79.9 | 82.5 | 79.9 | 82.5 | 5.0 |
| exchange_delivered_order_items | write | 416 | 76.2 | 95.0 | 76.2 | 95.0 | 76.2 | 95.0 | 3.8 |
| cancel_pending_order | write | 291 | 87.3 | 98.8 | 87.3 | 98.8 | 87.3 | 98.8 | 4.1 |
| modify_pending_order_address | write | 227 | 87.7 | 80.4 | 87.7 | 80.4 | 87.7 | 80.4 | 3.1 |
| modify_user_address | write | 119 | 84.0 | 100.0 | 84.0 | 100.0 | 84.0 | 100.0 | 2.5 |
| transfer_to_human_agents | write | 104 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| modify_pending_order_payment | write | 18 | 77.8 | 100.0 | 77.8 | 100.0 | 77.8 | 100.0 | 0.0 |

Read calls reusable (state-aware key) by trial index: t0: 67%, t1: 94%, t2: 95%, t3: 97%, t4: 98%, t5: 98%, t6: 97%, t7: 97%

**airline / gpt-4o** (200 episodes, 1164 tool calls, 4 trials; chronological = trial-major, so trial 0 is cold start)
| tool | kind | calls | (i) seen % | (i) equal % | (ii) seen % | (ii) equal % | (iii) seen % | (iii) equal % | seen only in other tasks % |
|---|---|---|---|---|---|---|---|---|---|
| calculate | pure | 96 | 13.5 | 100.0 | 13.5 | 100.0 | 13.5 | 100.0 | 1.0 |
| think | pure | 92 | 1.1 | 100.0 | 1.1 | 100.0 | 1.1 | 100.0 | 0.0 |
| get_reservation_details | read | 377 | 74.0 | 100.0 | 69.8 | 100.0 | 74.0 | 100.0 | 8.0 |
| search_direct_flight | read | 141 | 51.1 | 100.0 | 48.2 | 100.0 | 51.1 | 100.0 | 3.5 |
| get_user_details | read | 120 | 72.5 | 100.0 | 70.0 | 100.0 | 72.5 | 100.0 | 9.2 |
| search_onestop_flight | read | 38 | 57.9 | 100.0 | 55.3 | 100.0 | 57.9 | 100.0 | 5.3 |
| list_all_airports | read | 2 | 50.0 | 100.0 | 50.0 | 100.0 | 50.0 | 100.0 | 50.0 |
| update_reservation_flights | write | 104 | 39.4 | 100.0 | 39.4 | 100.0 | 39.4 | 100.0 | 1.0 |
| cancel_reservation | write | 69 | 62.3 | 100.0 | 62.3 | 100.0 | 62.3 | 100.0 | 4.3 |
| book_reservation | write | 53 | 15.1 | 75.0 | 15.1 | 75.0 | 15.1 | 75.0 | 0.0 |
| transfer_to_human_agents | write | 48 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| update_reservation_baggages | write | 14 | 21.4 | 100.0 | 21.4 | 100.0 | 21.4 | 100.0 | 0.0 |
| send_certificate | write | 8 | 50.0 | 100.0 | 50.0 | 100.0 | 50.0 | 100.0 | 12.5 |
| update_reservation_passengers | write | 2 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |

Read calls reusable (state-aware key) by trial index: t0: 16%, t1: 75%, t2: 86%, t3: 81%

**airline / sonnet-3.5** (400 episodes, 2761 tool calls, 8 trials; chronological = trial-major, so trial 0 is cold start)
| tool | kind | calls | (i) seen % | (i) equal % | (ii) seen % | (ii) equal % | (iii) seen % | (iii) equal % | seen only in other tasks % |
|---|---|---|---|---|---|---|---|---|---|
| think | pure | 101 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| calculate | pure | 44 | 29.5 | 100.0 | 29.5 | 100.0 | 29.5 | 100.0 | 2.3 |
| get_reservation_details | read | 1042 | 88.0 | 99.3 | 86.5 | 100.0 | 87.8 | 100.0 | 3.2 |
| search_direct_flight | read | 397 | 72.3 | 100.0 | 63.2 | 100.0 | 71.0 | 100.0 | 1.8 |
| get_user_details | read | 366 | 89.1 | 100.0 | 88.0 | 100.0 | 89.1 | 100.0 | 4.4 |
| search_onestop_flight | read | 170 | 65.9 | 100.0 | 56.5 | 100.0 | 65.3 | 100.0 | 1.8 |
| list_all_airports | read | 16 | 93.8 | 100.0 | 87.5 | 100.0 | 93.8 | 100.0 | 12.5 |
| update_reservation_flights | write | 183 | 60.7 | 88.3 | 60.7 | 88.3 | 60.7 | 88.3 | 3.8 |
| transfer_to_human_agents | write | 131 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| cancel_reservation | write | 119 | 75.6 | 97.8 | 75.6 | 97.8 | 75.6 | 97.8 | 4.2 |
| book_reservation | write | 96 | 35.4 | 91.2 | 35.4 | 91.2 | 35.4 | 91.2 | 1.0 |
| update_reservation_baggages | write | 52 | 71.2 | 81.1 | 71.2 | 81.1 | 71.2 | 81.1 | 5.8 |
| send_certificate | write | 22 | 54.5 | 100.0 | 54.5 | 100.0 | 54.5 | 100.0 | 4.5 |
| update_reservation_passengers | write | 22 | 90.9 | 50.0 | 90.9 | 50.0 | 90.9 | 50.0 | 4.5 |

Read calls reusable (state-aware key) by trial index: t0: 18%, t1: 80%, t2: 84%, t3: 82%, t4: 91%, t5: 90%, t6: 93%, t7: 99%

**airline / pooled** (600 episodes, 3925 tool calls, 8 trials; chronological = trial-major, so trial 0 is cold start)
| tool | kind | calls | (i) seen % | (i) equal % | (ii) seen % | (ii) equal % | (iii) seen % | (iii) equal % | seen only in other tasks % |
|---|---|---|---|---|---|---|---|---|---|
| think | pure | 193 | 0.5 | 100.0 | 0.5 | 100.0 | 0.5 | 100.0 | 0.0 |
| calculate | pure | 140 | 20.7 | 100.0 | 20.7 | 100.0 | 20.7 | 100.0 | 1.4 |
| get_reservation_details | read | 1419 | 91.1 | 99.5 | 88.9 | 100.0 | 90.9 | 100.0 | 2.4 |
| search_direct_flight | read | 538 | 74.5 | 100.0 | 67.5 | 100.0 | 73.6 | 100.0 | 2.0 |
| get_user_details | read | 486 | 91.8 | 100.0 | 89.9 | 100.0 | 91.8 | 100.0 | 3.1 |
| search_onestop_flight | read | 208 | 68.8 | 100.0 | 60.6 | 100.0 | 68.3 | 100.0 | 1.9 |
| list_all_airports | read | 18 | 94.4 | 100.0 | 88.9 | 100.0 | 94.4 | 100.0 | 16.7 |
| update_reservation_flights | write | 287 | 62.4 | 92.2 | 62.4 | 92.2 | 62.4 | 92.2 | 3.8 |
| cancel_reservation | write | 188 | 80.9 | 98.7 | 80.9 | 98.7 | 80.9 | 98.7 | 4.3 |
| transfer_to_human_agents | write | 179 | 0.0 | n/a | 0.0 | n/a | 0.0 | n/a | 0.0 |
| book_reservation | write | 149 | 30.9 | 87.0 | 30.9 | 87.0 | 30.9 | 87.0 | 0.7 |
| update_reservation_baggages | write | 66 | 68.2 | 71.1 | 68.2 | 71.1 | 68.2 | 71.1 | 4.5 |
| send_certificate | write | 30 | 66.7 | 100.0 | 66.7 | 100.0 | 66.7 | 100.0 | 3.3 |
| update_reservation_passengers | write | 24 | 91.7 | 54.5 | 91.7 | 54.5 | 91.7 | 54.5 | 4.2 |

Read calls reusable (state-aware key) by trial index: t0: 42%, t1: 83%, t2: 87%, t3: 90%, t4: 91%, t5: 93%, t6: 95%, t7: 99%