Regime W splits, 10 seeds; the outage hits one read tool for every test episode. Policies answer each call to the dead tool from the log. 'episodes served' = all calls to the dead tool answered; 'served and correct' = additionally every answer equal to the real return.

**retail / gpt-4o**, outage tools = ['get_order_details', 'get_user_details', 'find_user_id_by_name_zip']
| dead tool | policy | test calls | calls answered % | error % of answered | episodes calling it | episodes served % | served and all correct % |
|---|---|---|---|---|---|---|---|
| get_order_details | circuit breaker | 2600 | 0.0 | - | 1086 | 0.0 | 0.0 |
| get_order_details | plain last-value cache (input key) | 2600 | 98.5 | 0.7 | 1086 | 97.2 | 95.6 |
| get_order_details | state-aware cache, uncertified | 2600 | 94.8 | 0.0 | 1086 | 91.1 | 91.1 |
| get_order_details | certified state-aware cache @1% | 2600 | 0.0 | - | 1086 | 0.0 | 0.0 |
| get_order_details | certified input-keyed cache @1% | 2600 | 0.0 | - | 1086 | 0.0 | 0.0 |
| get_order_details | certified state-aware cache @2% | 2600 | 94.8 | 0.0 | 1086 | 91.1 | 91.1 |
| get_order_details | certified input-keyed cache @2% | 2600 | 98.5 | 0.7 | 1086 | 97.2 | 95.6 |
| get_user_details | circuit breaker | 987 | 0.0 | - | 987 | 0.0 | 0.0 |
| get_user_details | plain last-value cache (input key) | 987 | 100.0 | 0.0 | 987 | 100.0 | 100.0 |
| get_user_details | state-aware cache, uncertified | 987 | 100.0 | 0.0 | 987 | 100.0 | 100.0 |
| get_user_details | certified state-aware cache @1% | 987 | 0.0 | - | 987 | 0.0 | 0.0 |
| get_user_details | certified input-keyed cache @1% | 987 | 0.0 | - | 987 | 0.0 | 0.0 |
| get_user_details | certified state-aware cache @2% | 987 | 100.0 | 0.0 | 987 | 100.0 | 100.0 |
| get_user_details | certified input-keyed cache @2% | 987 | 100.0 | 0.0 | 987 | 100.0 | 100.0 |
| find_user_id_by_name_zip | circuit breaker | 960 | 0.0 | - | 928 | 0.0 | 0.0 |
| find_user_id_by_name_zip | plain last-value cache (input key) | 960 | 99.0 | 0.0 | 928 | 98.9 | 98.9 |
| find_user_id_by_name_zip | state-aware cache, uncertified | 960 | 99.0 | 0.0 | 928 | 98.9 | 98.9 |
| find_user_id_by_name_zip | certified state-aware cache @1% | 960 | 0.0 | - | 928 | 0.0 | 0.0 |
| find_user_id_by_name_zip | certified input-keyed cache @1% | 960 | 0.0 | - | 928 | 0.0 | 0.0 |
| find_user_id_by_name_zip | certified state-aware cache @2% | 960 | 99.0 | 0.0 | 928 | 98.9 | 98.9 |
| find_user_id_by_name_zip | certified input-keyed cache @2% | 960 | 99.0 | 0.0 | 928 | 98.9 | 98.9 |

**retail / sonnet-3.5**, outage tools = ['get_order_details', 'get_product_details', 'get_user_details']
| dead tool | policy | test calls | calls answered % | error % of answered | episodes calling it | episodes served % | served and all correct % |
|---|---|---|---|---|---|---|---|
| get_order_details | circuit breaker | 6159 | 0.0 | - | 2249 | 0.0 | 0.0 |
| get_order_details | plain last-value cache (input key) | 6159 | 99.9 | 0.7 | 2249 | 99.7 | 98.0 |
| get_order_details | state-aware cache, uncertified | 6159 | 97.4 | 0.0 | 2249 | 95.7 | 95.7 |
| get_order_details | certified state-aware cache @1% | 6159 | 97.4 | 0.0 | 2249 | 95.7 | 95.7 |
| get_order_details | certified input-keyed cache @1% | 6159 | 55.2 | 0.8 | 2249 | 51.0 | 49.9 |
| get_order_details | certified state-aware cache @2% | 6159 | 97.4 | 0.0 | 2249 | 95.7 | 95.7 |
| get_order_details | certified input-keyed cache @2% | 6159 | 99.9 | 0.7 | 2249 | 99.7 | 98.0 |
| get_product_details | circuit breaker | 2111 | 0.0 | - | 1405 | 0.0 | 0.0 |
| get_product_details | plain last-value cache (input key) | 2111 | 99.7 | 0.0 | 1405 | 99.6 | 99.6 |
| get_product_details | state-aware cache, uncertified | 2111 | 93.2 | 0.0 | 1405 | 92.0 | 92.0 |
| get_product_details | certified state-aware cache @1% | 2111 | 93.2 | 0.0 | 1405 | 92.0 | 92.0 |
| get_product_details | certified input-keyed cache @1% | 2111 | 66.3 | 0.0 | 1405 | 62.8 | 62.8 |
| get_product_details | certified state-aware cache @2% | 2111 | 93.2 | 0.0 | 1405 | 92.0 | 92.0 |
| get_product_details | certified input-keyed cache @2% | 2111 | 99.7 | 0.0 | 1405 | 99.6 | 99.6 |
| get_user_details | circuit breaker | 2120 | 0.0 | - | 2084 | 0.0 | 0.0 |
| get_user_details | plain last-value cache (input key) | 2120 | 100.0 | 0.7 | 2084 | 100.0 | 99.6 |
| get_user_details | state-aware cache, uncertified | 2120 | 99.5 | 0.0 | 2084 | 99.5 | 99.5 |
| get_user_details | certified state-aware cache @1% | 2120 | 99.5 | 0.0 | 2084 | 99.5 | 99.5 |
| get_user_details | certified input-keyed cache @1% | 2120 | 54.2 | 0.3 | 2084 | 54.3 | 54.2 |
| get_user_details | certified state-aware cache @2% | 2120 | 99.5 | 0.0 | 2084 | 99.5 | 99.5 |
| get_user_details | certified input-keyed cache @2% | 2120 | 100.0 | 0.7 | 2084 | 100.0 | 99.6 |

**retail / pooled**, outage tools = ['get_order_details', 'get_user_details', 'get_product_details']
| dead tool | policy | test calls | calls answered % | error % of answered | episodes calling it | episodes served % | served and all correct % |
|---|---|---|---|---|---|---|---|
| get_order_details | circuit breaker | 6159 | 0.0 | - | 2249 | 0.0 | 0.0 |
| get_order_details | plain last-value cache (input key) | 6159 | 99.9 | 0.7 | 2249 | 99.7 | 98.0 |
| get_order_details | state-aware cache, uncertified | 6159 | 98.7 | 0.0 | 2249 | 97.0 | 97.0 |
| get_order_details | certified state-aware cache @1% | 6159 | 98.7 | 0.0 | 2249 | 97.0 | 97.0 |
| get_order_details | certified input-keyed cache @1% | 6159 | 73.1 | 0.7 | 2249 | 67.0 | 65.7 |
| get_order_details | certified state-aware cache @2% | 6159 | 98.7 | 0.0 | 2249 | 97.0 | 97.0 |
| get_order_details | certified input-keyed cache @2% | 6159 | 99.9 | 0.7 | 2249 | 99.7 | 98.0 |
| get_user_details | circuit breaker | 2120 | 0.0 | - | 2084 | 0.0 | 0.0 |
| get_user_details | plain last-value cache (input key) | 2120 | 100.0 | 0.7 | 2084 | 100.0 | 99.6 |
| get_user_details | state-aware cache, uncertified | 2120 | 99.5 | 0.0 | 2084 | 99.5 | 99.5 |
| get_user_details | certified state-aware cache @1% | 2120 | 99.5 | 0.0 | 2084 | 99.5 | 99.5 |
| get_user_details | certified input-keyed cache @1% | 2120 | 72.2 | 0.3 | 2084 | 72.3 | 72.2 |
| get_user_details | certified state-aware cache @2% | 2120 | 99.5 | 0.0 | 2084 | 99.5 | 99.5 |
| get_user_details | certified input-keyed cache @2% | 2120 | 100.0 | 0.7 | 2084 | 100.0 | 99.6 |
| get_product_details | circuit breaker | 2111 | 0.0 | - | 1405 | 0.0 | 0.0 |
| get_product_details | plain last-value cache (input key) | 2111 | 99.7 | 0.0 | 1405 | 99.6 | 99.6 |
| get_product_details | state-aware cache, uncertified | 2111 | 95.0 | 0.0 | 1405 | 93.5 | 93.5 |
| get_product_details | certified state-aware cache @1% | 2111 | 95.0 | 0.0 | 1405 | 93.5 | 93.5 |
| get_product_details | certified input-keyed cache @1% | 2111 | 84.6 | 0.0 | 1405 | 81.7 | 81.7 |
| get_product_details | certified state-aware cache @2% | 2111 | 95.0 | 0.0 | 1405 | 93.5 | 93.5 |
| get_product_details | certified input-keyed cache @2% | 2111 | 99.7 | 0.0 | 1405 | 99.6 | 99.6 |

**airline / gpt-4o**, outage tools = ['get_reservation_details', 'search_direct_flight', 'get_user_details']
| dead tool | policy | test calls | calls answered % | error % of answered | episodes calling it | episodes served % | served and all correct % |
|---|---|---|---|---|---|---|---|
| get_reservation_details | circuit breaker | 947 | 0.0 | - | 384 | 0.0 | 0.0 |
| get_reservation_details | plain last-value cache (input key) | 947 | 91.1 | 0.0 | 384 | 90.6 | 90.6 |
| get_reservation_details | state-aware cache, uncertified | 947 | 84.8 | 0.0 | 384 | 86.5 | 86.5 |
| get_reservation_details | certified state-aware cache @1% | 947 | 0.0 | - | 384 | 0.0 | 0.0 |
| get_reservation_details | certified input-keyed cache @1% | 947 | 0.0 | - | 384 | 0.0 | 0.0 |
| get_reservation_details | certified state-aware cache @2% | 947 | 0.0 | - | 384 | 0.0 | 0.0 |
| get_reservation_details | certified input-keyed cache @2% | 947 | 0.0 | - | 384 | 0.0 | 0.0 |
| search_direct_flight | circuit breaker | 362 | 0.0 | - | 169 | 0.0 | 0.0 |
| search_direct_flight | plain last-value cache (input key) | 362 | 80.1 | 0.0 | 169 | 75.7 | 75.7 |
| search_direct_flight | state-aware cache, uncertified | 362 | 71.5 | 0.0 | 169 | 63.3 | 63.3 |
| search_direct_flight | certified state-aware cache @1% | 362 | 0.0 | - | 169 | 0.0 | 0.0 |
| search_direct_flight | certified input-keyed cache @1% | 362 | 0.0 | - | 169 | 0.0 | 0.0 |
| search_direct_flight | certified state-aware cache @2% | 362 | 0.0 | - | 169 | 0.0 | 0.0 |
| search_direct_flight | certified input-keyed cache @2% | 362 | 0.0 | - | 169 | 0.0 | 0.0 |
| get_user_details | circuit breaker | 306 | 0.0 | - | 306 | 0.0 | 0.0 |
| get_user_details | plain last-value cache (input key) | 306 | 93.1 | 0.0 | 306 | 93.1 | 93.1 |
| get_user_details | state-aware cache, uncertified | 306 | 90.5 | 0.0 | 306 | 90.5 | 90.5 |
| get_user_details | certified state-aware cache @1% | 306 | 0.0 | - | 306 | 0.0 | 0.0 |
| get_user_details | certified input-keyed cache @1% | 306 | 0.0 | - | 306 | 0.0 | 0.0 |
| get_user_details | certified state-aware cache @2% | 306 | 0.0 | - | 306 | 0.0 | 0.0 |
| get_user_details | certified input-keyed cache @2% | 306 | 0.0 | - | 306 | 0.0 | 0.0 |

**airline / sonnet-3.5**, outage tools = ['get_reservation_details', 'search_direct_flight', 'get_user_details']
| dead tool | policy | test calls | calls answered % | error % of answered | episodes calling it | episodes served % | served and all correct % |
|---|---|---|---|---|---|---|---|
| get_reservation_details | circuit breaker | 2663 | 0.0 | - | 930 | 0.0 | 0.0 |
| get_reservation_details | plain last-value cache (input key) | 2663 | 98.3 | 0.0 | 930 | 97.6 | 97.6 |
| get_reservation_details | state-aware cache, uncertified | 2663 | 97.5 | 0.0 | 930 | 96.7 | 96.7 |
| get_reservation_details | certified state-aware cache @1% | 2663 | 0.0 | - | 930 | 0.0 | 0.0 |
| get_reservation_details | certified input-keyed cache @1% | 2663 | 0.0 | - | 930 | 0.0 | 0.0 |
| get_reservation_details | certified state-aware cache @2% | 2663 | 97.5 | 0.0 | 930 | 96.7 | 96.7 |
| get_reservation_details | certified input-keyed cache @2% | 2663 | 98.3 | 0.0 | 930 | 97.6 | 97.6 |
| search_direct_flight | circuit breaker | 967 | 0.0 | - | 403 | 0.0 | 0.0 |
| search_direct_flight | plain last-value cache (input key) | 967 | 94.1 | 0.0 | 403 | 93.1 | 93.1 |
| search_direct_flight | state-aware cache, uncertified | 967 | 87.4 | 0.0 | 403 | 88.3 | 88.3 |
| search_direct_flight | certified state-aware cache @1% | 967 | 0.0 | - | 403 | 0.0 | 0.0 |
| search_direct_flight | certified input-keyed cache @1% | 967 | 0.0 | - | 403 | 0.0 | 0.0 |
| search_direct_flight | certified state-aware cache @2% | 967 | 87.4 | 0.0 | 403 | 88.3 | 88.3 |
| search_direct_flight | certified input-keyed cache @2% | 967 | 94.1 | 0.0 | 403 | 93.1 | 93.1 |
| get_user_details | circuit breaker | 946 | 0.0 | - | 889 | 0.0 | 0.0 |
| get_user_details | plain last-value cache (input key) | 946 | 98.5 | 0.0 | 889 | 98.4 | 98.4 |
| get_user_details | state-aware cache, uncertified | 946 | 97.7 | 0.0 | 889 | 97.5 | 97.5 |
| get_user_details | certified state-aware cache @1% | 946 | 0.0 | - | 889 | 0.0 | 0.0 |
| get_user_details | certified input-keyed cache @1% | 946 | 0.0 | - | 889 | 0.0 | 0.0 |
| get_user_details | certified state-aware cache @2% | 946 | 97.7 | 0.0 | 889 | 97.5 | 97.5 |
| get_user_details | certified input-keyed cache @2% | 946 | 98.5 | 0.0 | 889 | 98.4 | 98.4 |

**airline / pooled**, outage tools = ['get_reservation_details', 'search_direct_flight', 'get_user_details']
| dead tool | policy | test calls | calls answered % | error % of answered | episodes calling it | episodes served % | served and all correct % |
|---|---|---|---|---|---|---|---|
| get_reservation_details | circuit breaker | 2663 | 0.0 | - | 930 | 0.0 | 0.0 |
| get_reservation_details | plain last-value cache (input key) | 2663 | 99.2 | 0.0 | 930 | 99.2 | 99.2 |
| get_reservation_details | state-aware cache, uncertified | 2663 | 99.2 | 0.0 | 930 | 99.2 | 99.2 |
| get_reservation_details | certified state-aware cache @1% | 2663 | 0.0 | - | 930 | 0.0 | 0.0 |
| get_reservation_details | certified input-keyed cache @1% | 2663 | 0.0 | - | 930 | 0.0 | 0.0 |
| get_reservation_details | certified state-aware cache @2% | 2663 | 99.2 | 0.0 | 930 | 99.2 | 99.2 |
| get_reservation_details | certified input-keyed cache @2% | 2663 | 99.2 | 0.0 | 930 | 99.2 | 99.2 |
| search_direct_flight | circuit breaker | 967 | 0.0 | - | 403 | 0.0 | 0.0 |
| search_direct_flight | plain last-value cache (input key) | 967 | 94.1 | 0.0 | 403 | 93.1 | 93.1 |
| search_direct_flight | state-aware cache, uncertified | 967 | 89.0 | 0.0 | 403 | 91.3 | 91.3 |
| search_direct_flight | certified state-aware cache @1% | 967 | 0.0 | - | 403 | 0.0 | 0.0 |
| search_direct_flight | certified input-keyed cache @1% | 967 | 0.0 | - | 403 | 0.0 | 0.0 |
| search_direct_flight | certified state-aware cache @2% | 967 | 89.0 | 0.0 | 403 | 91.3 | 91.3 |
| search_direct_flight | certified input-keyed cache @2% | 967 | 94.1 | 0.0 | 403 | 93.1 | 93.1 |
| get_user_details | circuit breaker | 946 | 0.0 | - | 889 | 0.0 | 0.0 |
| get_user_details | plain last-value cache (input key) | 946 | 98.5 | 0.0 | 889 | 98.4 | 98.4 |
| get_user_details | state-aware cache, uncertified | 946 | 97.7 | 0.0 | 889 | 97.5 | 97.5 |
| get_user_details | certified state-aware cache @1% | 946 | 0.0 | - | 889 | 0.0 | 0.0 |
| get_user_details | certified input-keyed cache @1% | 946 | 0.0 | - | 889 | 0.0 | 0.0 |
| get_user_details | certified state-aware cache @2% | 946 | 97.7 | 0.0 | 889 | 97.5 | 97.5 |
| get_user_details | certified input-keyed cache @2% | 946 | 98.5 | 0.0 | 889 | 98.4 | 98.4 |