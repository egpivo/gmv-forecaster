AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# Based on summary statistics
STORE_GMV_BINS = {
    "last_month_store_gmv": [0, 274, 1259, 4247, 906552],
    "last_quarter_store_gmv": [0, 2994.75, 12169, 36205, 907924],
    "last_half_year_store_gmv": [0, 6977, 21542, 66701, 1885077],
    "last_year_store_gmv": [0, 13762, 35850, 115723, 1894692],
}


USER_GMV_BINS = {
    "last_month_user_gmv": [0, 1.182400e04, 1.895800e04, 2.896000e04, 9.100910e05],
    "last_quarter_user_gmv": [0, 4.337700e04, 6.076500e04, 8.433300e04, 9.243240e05],
    "last_half_year_user_gmv": [0, 8.853825e04, 1.212690e05, 1.633340e05, 1.973215e06],
    "last_year_user_gmv": [0, 1.701480e05, 2.331470e05, 3.108270e05, 2.073184e06],
}
GMV_BINS = {"user": USER_GMV_BINS, "store": STORE_GMV_BINS}


STORE_PURCHASE_BINS = {
    "last_month_store_purchase": [
        0,
        1.000000e00,
        2.000000e00,
        1.600000e01,
        1.120000e02,
    ],
    "last_quarter_store_purchase": [
        0,
        2.000000e00,
        5.000000e00,
        4.500000e01,
        3.070000e02,
    ],
    "last_half_year_store_purchase": [
        0,
        5.000000e00,
        9.000000e00,
        8.300000e01,
        5.640000e02,
    ],
    "last_year_store_purchase": [0, 8.000000e00, 1.500000e01, 1.450000e02, 1.022000e03],
}


USER_PURCHASE_BINS = {
    "last_month_user_purchase": [0, 1.000000e00, 1.300000e01, 1.700000e01, 3.400000e01],
    "last_quarter_user_purchase": [
        0,
        3.100000e01,
        4.000000e01,
        4.900000e01,
        7.800000e01,
    ],
    "last_half_year_user_purchase": [
        0,
        6.100000e01,
        7.700000e01,
        9.300000e01,
        1.380000e02,
    ],
    "last_year_user_purchase": [0, 1.140000e02, 1.470000e02, 1.790000e02, 2.500000e02],
}
PURCHASE_BINS = {"user": USER_PURCHASE_BINS, "store": STORE_PURCHASE_BINS}


USER_RECENCY_BINS = [0, 1, 4, 730]
STORE_RECENCY_BINS = [0, 12, 37, 95, 730]
RECENCY_BINS = {"user": USER_RECENCY_BINS, "store": STORE_RECENCY_BINS}


TRANSACTIONS_AGE_BINS = [0, 1.160000e02, 2.660000e02, 4.550000e02, 7.300000e02]
