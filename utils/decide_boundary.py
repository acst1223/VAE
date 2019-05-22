def decide_boundary(ll_n, ll_a):
    '''
    When evaluating, we have two lists of ll (log likelihood):
    log likelihood according to normal and anomalous instances.
    What we need to do is to decide a boundary value:
    Instances with ll above this boundary value will be considered normal,
    while the others will be considered anomalous.
    The best boundary should be the one that would yield a best F1-Score.

    Args:
        ll_n: List of log likelihood of normal instances.
        ll_a: List of log likelihood of anomalous instances.

    Returns:
        boundary: A float that would yield a best F1-Score.
        precision according to the boundary
        recall according to the boundary
        F1-Score according to the boundary
    '''
    l_n, l_a = len(ll_n), len(ll_a)
    assert l_a > 0
    ll_n = sorted(ll_n)
    ll_a = sorted(ll_a)

    # Assume that the number of normal instances with ll <= boundary is k',
    # the number of anomalous instances with ll <= boundary is k,
    # and the total number of anomalous instances is l_a,
    # then maximize F1-Score is equivalent to minimizing (l_a + k') / k.
    #
    # In this function, k' == i_n, k == i_a + 1
    i_n = 0
    m = float('inf')
    boundary, precision, recall, f1_score = None, None, None, None
    for i_a in range(l_a):
        while i_n < l_n and ll_n[i_n] <= ll_a[i_a]:
            i_n += 1
        m_tmp = (l_a + i_n) / (i_a + 1)
        if m_tmp < m:
            m = m_tmp
            boundary = ll_a[i_a]
            precision = (i_a + 1) / (i_n + i_a + 1)
            recall = (i_a + 1) / l_a
            f1_score = 2 * precision * recall / (precision + recall)
    return boundary, precision, recall, f1_score
