# Task 1 Experiment Results

- Updated: 2026-05-13T20:01:03.990234+08:00
- Records scanned: 374
- Sort: competition_score_proxy desc, then mse asc.

| Rank | Run | Model | Proxy | MSE | Forecast MSE | Long MSE | Segment3 RMSE | Status | Params |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | task1-zoo-medium-finetuned-fno/temporal_tail_blend_deeponet_lite | temporal_tail_blend_deeponet_lite | 58.9942038 | 0.001526509685 | 0.0016068523 | 0.0008058132656 | 0.02838685022 | ok | {"base_name": "fno_ensemble", "cut": 120, "kind": "temporal_tail_blend", "tail_name": "deeponet_lite", "tail_weight": 0.6000000000000001} |
| 2 | autonomous/84d1ed78b3164c30a00c0bd2debec59f/current-final-proxy | current-final-proxy | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok |  |
| 3 | task1-autonomous-bootstrap-recorded/nodes/autonomous/1dc7d01ea33e4d3f92ba950cf53fd3f7/current-final-proxy | current-final-proxy | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok |  |
| 4 | task1-autonomous-bootstrap-smoke2/nodes/autonomous/4224229a4ebe4cfba002913d586f4016/current-final-proxy | current-final-proxy | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok |  |
| 5 | task1-autonomous-local-grid-r2/nodes/autonomous/a3e26b09b8874699ab4b862758eb1042/current-final-proxy | current-final-proxy | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok |  |
| 6 | task1-finetune-nu0.1-short-proxy-weight-search/rank-13-mse-0.0016034222 | rank-13-mse-0.0016034222 | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok |  |
| 7 | task1-zoo-medium-finetuned-fno/cluster_em_ensemble | cluster_em_ensemble | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok | {"cluster_weights": {"0": {"deeponet_lite": 0.0, "fno_ensemble": 1.0, "residual_refiner": 0.0}, "1": {"deeponet_lite": 0.0, "fno_ensemble": 1.0, "residual_refiner": 0.0}, "2": {"deeponet_lite": 0.0, "fno_ensemble": 1.... |
| 8 | task1-zoo-medium-finetuned-fno/fno_ensemble | fno_ensemble | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok |  |
| 9 | task1-zoo-medium-finetuned-fno/global_ensemble | global_ensemble | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok | {"grid_step": 0.05, "kind": "global_convex", "weights": {"deeponet_lite": 0.0, "fno_ensemble": 1.0, "residual_refiner": 0.0}} |
| 10 | task1-zoo-medium-finetuned-fno/temporal_tail_blend_residual_refiner | temporal_tail_blend_residual_refiner | 58.18577979 | 0.001603422098 | 0.001687812735 | 0.000967734135 | 0.03110842547 | ok | {"base_name": "fno_ensemble", "cut": 105, "kind": "temporal_tail_blend", "tail_name": "residual_refiner", "tail_weight": 0.0} |
| 11 | task1-finetune-nu0.1-short-proxy-weight-search/rank-06-mse-0.0016028347 | rank-06-mse-0.0016028347 | 58.18471695 | 0.00160283455 | 0.001687194263 | 0.0009633218514 | 0.03103742662 | ok |  |
| 12 | task1-finetune-nu0.1-short-proxy-weight-search/rank-22-mse-0.0016042029 | rank-22-mse-0.0016042029 | 58.1842187 | 0.001604202749 | 0.001688634473 | 0.0009724743633 | 0.03118452121 | ok |  |
| 13 | task1-finetune-nu0.1-short-proxy-weight-search/rank-23-mse-0.0016043964 | rank-23-mse-0.0016043964 | 58.18217195 | 0.001604396243 | 0.00168883815 | 0.0009679554638 | 0.03111198264 | ok |  |
| 14 | task1-finetune-nu0.1-short-proxy-weight-search/rank-16-mse-0.0016037386 | rank-16-mse-0.0016037386 | 58.18214504 | 0.001603738429 | 0.001688145715 | 0.0009634103076 | 0.03103885158 | ok |  |
| 15 | task1-finetune-nu0.1-short-proxy-weight-search/rank-04-mse-0.0016024402 | rank-04-mse-0.0016024402 | 58.18101583 | 0.001602440136 | 0.00168677909 | 0.0009592375518 | 0.03097156037 | ok |  |
| 16 | task1-finetune-nu0.1-short-proxy-weight-search/rank-35-mse-0.0016051766 | rank-35-mse-0.0016051766 | 58.18004921 | 0.001605176464 | 0.001689659436 | 0.0009775424967 | 0.03126567602 | ok |  |
| 17 | task1-finetune-nu0.1-short-proxy-weight-search/rank-36-mse-0.0016052473 | rank-36-mse-0.0016052473 | 58.17957951 | 0.001605247154 | 0.001689733847 | 0.0009728285618 | 0.03119019977 | ok |  |
| 18 | task1-finetune-nu0.1-short-proxy-weight-search/rank-10-mse-0.0016032738 | rank-10-mse-0.0016032738 | 58.1794839 | 0.001603273752 | 0.001687656581 | 0.0009591931356 | 0.03097084331 | ok |  |
| 19 | task1-finetune-nu0.1-short-proxy-weight-search/rank-29-mse-0.0016047101 | rank-29-mse-0.0016047101 | 58.17879639 | 0.001604709993 | 0.001689168414 | 0.0009635725483 | 0.03104146498 | ok |  |
| 20 | task1-finetune-nu0.1-short-proxy-weight-search/rank-41-mse-0.0016054382 | rank-41-mse-0.0016054382 | 58.1777878 | 0.001605438069 | 0.00168993481 | 0.0009682505758 | 0.03111672502 | ok |  |
| 21 | task1-finetune-nu0.1-short-proxy-weight-search/rank-21-mse-0.0016041752 | rank-21-mse-0.0016041752 | 58.17717505 | 0.001604175054 | 0.00168860532 | 0.0009592225066 | 0.03097131748 | ok |  |
| 22 | task1-finetune-nu0.1-short-proxy-weight-search/rank-45-mse-0.0016057494 | rank-45-mse-0.0016057494 | 58.17467152 | 0.001605749242 | 0.00169026236 | 0.000963808575 | 0.03104526655 | ok |  |
| 23 | task1-finetune-nu0.1-short-proxy-weight-search/rank-02-mse-0.0016022389 | rank-02-mse-0.0016022389 | 58.17466403 | 0.001602238788 | 0.001686567146 | 0.0009554811544 | 0.0309108582 | ok |  |
| 24 | task1-finetune-nu0.1-short-proxy-weight-search/rank-52-mse-0.0016062913 | rank-52-mse-0.0016062913 | 58.17438371 | 0.001606291134 | 0.001690832773 | 0.0009780295654 | 0.03127346424 | ok |  |
| 25 | task1-finetune-nu0.1-short-proxy-weight-search/rank-08-mse-0.0016030022 | rank-08-mse-0.0016030022 | 58.17417558 | 0.001603002142 | 0.001687370676 | 0.0009553038711 | 0.03090799041 | ok |  |
| 26 | task1-finetune-nu0.1-short-proxy-weight-search/rank-55-mse-0.0016063594 | rank-55-mse-0.0016063594 | 58.17416466 | 0.001606359247 | 0.00169090447 | 0.0009732565436 | 0.03119705986 | ok |  |
| 27 | task1-finetune-nu0.1-short-proxy-weight-search/rank-34-mse-0.0016051442 | rank-34-mse-0.0016051442 | 58.17408967 | 0.00160514404 | 0.001689625305 | 0.0009593256611 | 0.03097298276 | ok |  |
| 28 | task1-finetune-nu0.1-short-proxy-weight-search/rank-54-mse-0.0016063435 | rank-54-mse-0.0016063435 | 58.17328723 | 0.001606343319 | 0.001690887704 | 0.0009829386145 | 0.03135185185 | ok |  |
| 29 | task1-finetune-nu0.1-short-proxy-weight-search/rank-17-mse-0.0016038333 | rank-17-mse-0.0016038333 | 58.1729103 | 0.00160383318 | 0.001688245452 | 0.000955200371 | 0.03090631604 | ok |  |
| 30 | task1-finetune-nu0.1-short-proxy-weight-search/rank-59-mse-0.0016065477 | rank-59-mse-0.0016065477 | 58.17262798 | 0.001606547582 | 0.001691102718 | 0.0009686194699 | 0.03112265204 | ok |  |
| 31 | task1-finetune-nu0.1-short-proxy-weight-search/rank-30-mse-0.001604732 | rank-30-mse-0.001604732 | 58.17086847 | 0.0016047319 | 0.001689191474 | 0.0009551706518 | 0.03090583524 | ok |  |
| 32 | task1-finetune-nu0.1-short-proxy-weight-search/rank-47-mse-0.0016061808 | rank-47-mse-0.0016061808 | 58.17022826 | 0.001606180709 | 0.001690716536 | 0.0009595025997 | 0.03097583897 | ok |  |
| 33 | task1-finetune-nu0.1-short-proxy-weight-search/rank-62-mse-0.0016068563 | rank-62-mse-0.0016068563 | 58.16977105 | 0.001606856176 | 0.001691427553 | 0.0009641183833 | 0.03105025577 | ok |  |
| 34 | task1-finetune-nu0.1-short-proxy-weight-search/rank-43-mse-0.0016056984 | rank-43-mse-0.0016056984 | 58.1680504 | 0.001605698309 | 0.001690208746 | 0.0009552147211 | 0.03090654819 | ok |  |
| 35 | task1-finetune-nu0.1-short-proxy-weight-search/rank-74-mse-0.0016075392 | rank-74-mse-0.0016075392 | 58.16797284 | 0.00160753934 | 0.001692146674 | 0.0009737587494 | 0.03120510775 | ok |  |
| 36 | task1-finetune-nu0.1-short-proxy-weight-search/rank-72-mse-0.0016074737 | rank-72-mse-0.0016074737 | 58.16794133 | 0.001607473813 | 0.001692077698 | 0.0009785908688 | 0.03128243707 | ok |  |
| 37 | task1-finetune-nu0.1-short-proxy-weight-search/rank-78-mse-0.0016077249 | rank-78-mse-0.0016077249 | 58.16669118 | 0.00160772509 | 0.0016923422 | 0.0009690625788 | 0.03112976998 | ok |  |
| 38 | task1-finetune-nu0.1-short-proxy-weight-search/rank-73-mse-0.0016075284 | rank-73-mse-0.0016075284 | 58.1665988 | 0.00160752858 | 0.001692135348 | 0.0009835590139 | 0.03136174443 | ok |  |
| 39 | task1-finetune-nu0.1-short-proxy-weight-search/rank-07-mse-0.0016029237 | rank-07-mse-0.0016029237 | 58.16620767 | 0.00160292363 | 0.001687288032 | 0.0009517425469 | 0.03085032491 | ok |  |
| 40 | task1-finetune-nu0.1-short-proxy-weight-search/rank-14-mse-0.0016036845 | rank-14-mse-0.0016036845 | 58.16598919 | 0.001603684407 | 0.001688088849 | 0.0009515061794 | 0.03084649379 | ok |  |
| 41 | task1-autonomous-local-grid-r2/nodes/autonomous/a3e26b09b8874699ab4b862758eb1042/grid-delta-neg0.010 | grid-delta-neg0.010 | 58.16564964 | 0.00160223054 | 0.001686558463 | 0.0009520526992 | 0.03085535122 | ok |  |
| 42 | task1-finetune-nu0.1-short-proxy-weight-search/rank-01-mse-0.0016022306 | rank-01-mse-0.0016022306 | 58.16564964 | 0.00160223054 | 0.001686558463 | 0.0009520526992 | 0.03085535122 | ok |  |
| 43 | task1-finetune-nu0.1-short-proxy-weight-search/rank-69-mse-0.0016072852 | rank-69-mse-0.0016072852 | 58.1655914 | 0.001607285063 | 0.001691879013 | 0.0009597533193 | 0.03097988572 | ok |  |
| 44 | task1-finetune-nu0.1-short-proxy-weight-search/rank-26-mse-0.001604513 | rank-26-mse-0.001604513 | 58.16499439 | 0.001604512864 | 0.00168896091 | 0.0009513435911 | 0.03084385824 | ok |  |
| 45 | task1-finetune-nu0.1-short-proxy-weight-search/rank-60-mse-0.0016067325 | rank-60-mse-0.0016067325 | 58.16445657 | 0.001606732398 | 0.001691297261 | 0.0009553325704 | 0.03090845468 | ok |  |
| 46 | task1-finetune-nu0.1-short-proxy-weight-search/rank-82-mse-0.001608031 | rank-82-mse-0.001608031 | 58.16409367 | 0.001608031098 | 0.001692664314 | 0.0009645023929 | 0.03105643883 | ok |  |
| 47 | task1-autonomous-local-grid-r2/nodes/autonomous/a3e26b09b8874699ab4b862758eb1042/grid-delta-pos0.010 | grid-delta-pos0.010 | 58.16394831 | 0.001607703608 | 0.001692319587 | 0.000988663147 | 0.03144301428 | ok |  |
| 48 | task1-finetune-nu0.1-short-proxy-weight-search/rank-77-mse-0.0016077034 | rank-77-mse-0.0016077034 | 58.16394831 | 0.001607703608 | 0.001692319587 | 0.000988663147 | 0.03144301428 | ok |  |
| 49 | task1-finetune-nu0.1-short-proxy-weight-search/rank-38-mse-0.0016054091 | rank-38-mse-0.0016054091 | 58.16322343 | 0.001605409009 | 0.00168990422 | 0.0009512547883 | 0.03084241865 | ok |  |
| 50 | task1-finetune-nu0.1-short-proxy-weight-search/rank-95-mse-0.0016087867 | rank-95-mse-0.0016087867 | 58.16100908 | 0.0016087868 | 0.00169345979 | 0.0009743342984 | 0.03121432842 | ok |  |
| 51 | task1-finetune-nu0.1-short-proxy-weight-search/rank-93-mse-0.0016087237 | rank-93-mse-0.0016087237 | 58.16072719 | 0.001608723853 | 0.00169339353 | 0.0009792255066 | 0.0312925791 | ok |  |
| 52 | task1-finetune-nu0.1-short-proxy-weight-search/rank-56-mse-0.001606373 | rank-56-mse-0.001606373 | 58.16067664 | 0.001606372836 | 0.001690918775 | 0.0009512397679 | 0.03084217515 | ok |  |
| 53 | task1-finetune-nu0.1-short-proxy-weight-search/rank-91-mse-0.0016084572 | rank-91-mse-0.0016084572 | 58.16017775 | 0.001608457404 | 0.001693113057 | 0.0009600782355 | 0.03098512926 | ok |  |
| 54 | task1-finetune-nu0.1-short-proxy-weight-search/rank-81-mse-0.0016078343 | rank-81-mse-0.0016078343 | 58.16008747 | 0.001607834176 | 0.001692457028 | 0.0009555242083 | 0.03091155461 | ok |  |
| 55 | task1-finetune-nu0.1-short-proxy-weight-search/rank-100-mse-0.0016089698 | rank-100-mse-0.0016089698 | 58.15998232 | 0.001608969971 | 0.001693652601 | 0.0009695790398 | 0.03113806416 | ok |  |
| 56 | task1-finetune-nu0.1-short-proxy-weight-search/rank-94-mse-0.0016087811 | rank-94-mse-0.0016087811 | 58.15913889 | 0.0016087812 | 0.001693453895 | 0.0009842527386 | 0.03137280253 | ok |  |
| 57 | task1-finetune-nu0.1-short-proxy-weight-search/rank-108-mse-0.0016092733 | rank-108-mse-0.0016092733 | 58.15764414 | 0.001609273404 | 0.001693972004 | 0.0009649597716 | 0.03106380163 | ok |  |
| 58 | task1-finetune-nu0.1-short-proxy-weight-search/rank-70-mse-0.0016074045 | rank-70-mse-0.0016074045 | 58.1573544 | 0.001607404349 | 0.001692004578 | 0.0009512985336 | 0.03084312782 | ok |  |
| 59 | task1-finetune-nu0.1-short-proxy-weight-search/rank-25-mse-0.001604487 | rank-25-mse-0.001604487 | 58.15645518 | 0.001604486965 | 0.001688933647 | 0.0009478445091 | 0.03078708348 | ok |  |
| 60 | task1-finetune-nu0.1-short-proxy-weight-search/rank-15-mse-0.0016037288 | rank-15-mse-0.0016037288 | 58.1564 | 0.001603728769 | 0.001688135546 | 0.000948139968 | 0.03079188153 | ok |  |
| 61 | task1-finetune-nu0.1-short-proxy-weight-search/rank-99-mse-0.0016089587 | rank-99-mse-0.0016089587 | 58.15624726 | 0.001608958806 | 0.001693640848 | 0.0009894159573 | 0.03145498303 | ok |  |
| 62 | task1-finetune-nu0.1-short-proxy-weight-search/rank-37-mse-0.0016053129 | rank-37-mse-0.0016053129 | 58.15573455 | 0.001605312845 | 0.001689802994 | 0.0009476228358 | 0.03078348317 | ok |  |
| 63 | task1-finetune-nu0.1-short-proxy-weight-search/rank-09-mse-0.0016030383 | rank-09-mse-0.0016030383 | 58.15556895 | 0.001603038256 | 0.001687408691 | 0.0009485092044 | 0.03079787662 | ok |  |
| 64 | task1-finetune-nu0.1-short-proxy-weight-search/rank-101-mse-0.0016090038 | rank-101-mse-0.0016090038 | 58.15494184 | 0.001609003928 | 0.001693688346 | 0.0009557900228 | 0.03091585391 | ok |  |
| 65 | task1-finetune-nu0.1-short-proxy-weight-search/rank-49-mse-0.0016062065 | rank-49-mse-0.0016062065 | 58.15423829 | 0.001606206409 | 0.001690743588 | 0.000947474944 | 0.03078108094 | ok |  |
| 66 | task1-finetune-nu0.1-short-proxy-weight-search/rank-114-mse-0.001609697 | rank-114-mse-0.001609697 | 58.15399199 | 0.001609697126 | 0.001694418027 | 0.0009604765237 | 0.03099155568 | ok |  |
| 67 | task1-finetune-nu0.1-short-proxy-weight-search/rank-03-mse-0.0016024155 | rank-03-mse-0.0016024155 | 58.15396195 | 0.001602415432 | 0.001686753086 | 0.000948952236 | 0.03080506835 | ok |  |
| 68 | task1-finetune-nu0.1-short-proxy-weight-search/rank-121-mse-0.0016101018 | rank-121-mse-0.0016101018 | 58.15327221 | 0.001610101949 | 0.001694844156 | 0.0009749836405 | 0.03122472803 | ok |  |
| 69 | task1-finetune-nu0.1-short-proxy-weight-search/rank-92-mse-0.0016085037 | rank-92-mse-0.0016085037 | 58.15325722 | 0.001608503543 | 0.001693161624 | 0.0009514310789 | 0.03084527644 | ok |  |
| 70 | task1-finetune-nu0.1-short-proxy-weight-search/rank-118-mse-0.0016100415 | rank-118-mse-0.0016100415 | 58.15274015 | 0.001610041582 | 0.001694780613 | 0.0009799339354 | 0.03130389649 | ok |  |
| 71 | task1-finetune-nu0.1-short-proxy-weight-search/rank-127-mse-0.0016102824 | rank-127-mse-0.0016102824 | 58.1525002 | 0.001610282544 | 0.001695034257 | 0.0009701692976 | 0.0311475408 | ok |  |
| 72 | task1-finetune-nu0.1-short-proxy-weight-search/rank-106-mse-0.0016092565 | rank-106-mse-0.0016092565 | 58.15205533 | 0.001609256669 | 0.001693954389 | 0.0009947151631 | 0.0315391053 | ok |  |
| 73 | task1-finetune-nu0.1-short-proxy-weight-search/rank-66-mse-0.0016071678 | rank-66-mse-0.0016071678 | 58.15196663 | 0.001607167659 | 0.001691755431 | 0.0009474008398 | 0.03077987719 | ok |  |
| 74 | task1-finetune-nu0.1-short-proxy-weight-search/rank-120-mse-0.0016101014 | rank-120-mse-0.0016101014 | 58.15090634 | 0.001610101507 | 0.001694843692 | 0.0009850202539 | 0.03138503232 | ok |  |
| 75 | task1-finetune-nu0.1-short-proxy-weight-search/rank-124-mse-0.0016102409 | rank-124-mse-0.0016102409 | 58.14902415 | 0.001610241076 | 0.001694990606 | 0.0009561292301 | 0.0309213394 | ok |  |
| 76 | task1-finetune-nu0.1-short-proxy-weight-search/rank-87-mse-0.0016081967 | rank-87-mse-0.0016081967 | 58.14891998 | 0.001608196589 | 0.001692838515 | 0.0009474005148 | 0.03077987191 | ok |  |
| 77 | task1-finetune-nu0.1-short-proxy-weight-search/rank-113-mse-0.0016096706 | rank-113-mse-0.0016096706 | 58.14838375 | 0.001609670716 | 0.001694390228 | 0.0009516378046 | 0.03084862727 | ok |  |
| 78 | task1-finetune-nu0.1-short-proxy-weight-search/rank-126-mse-0.0016102815 | rank-126-mse-0.0016102815 | 58.1477739 | 0.001610281692 | 0.00169503336 | 0.0009902425603 | 0.03146811975 | ok |  |
| 79 | task1-finetune-nu0.1-short-proxy-weight-search/rank-39-mse-0.0016054098 | rank-39-mse-0.0016054098 | 58.14557296 | 0.001605409748 | 0.001689904998 | 0.0009443187926 | 0.03072977046 | ok |  |
| 80 | task1-finetune-nu0.1-short-proxy-weight-search/rank-28-mse-0.0016046542 | rank-28-mse-0.0016046542 | 58.1452406 | 0.001604654131 | 0.001689109611 | 0.0009446733363 | 0.03073553865 | ok |  |
