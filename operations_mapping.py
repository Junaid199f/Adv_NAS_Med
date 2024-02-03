operations_mapping = { 
    0:'max_pool_3x3',
    1:'avg_pool_3x3',
    2:'skip_connect',
    3:'sep_conv_3x3',
    4:'sep_conv_5x5',
    5:'dil_conv_3x3',
    6:'dil_conv_5x5',
    7:'conv_7x1_1x7',
    8:'inv_res_3x3',
    9:'inv_res_5x5',
    10:'blur_pool_3x3',
    11:'octave_conv_5x5',
    12:'octave_conv_3x3',
    13: 'mbconv_k7_t1',
    14:'mbconv_k5_t1',
    15:'mbconv_k3_t1'}

primitives = [
    'max_pool_3x3', 'avg_pool_3x3',
    'skip_connect', 'sep_conv_3x3',
    'sep_conv_5x5', 'dil_conv_3x3',
    'dil_conv_5x5', 'conv_7x1_1x7',
    'inv_res_3x3', 'inv_res_5x5',
    'octave_conv_5x5','octave_conv_3x3',
    'mbconv_k7_t1','mbconv_k5_t1',
    'mbconv_k3_t1'
    ]

attentions = {
0:'Identity',
1:'SE',
2:'BAM',
3:'CBAM',
4:'GE_theta-plus',
5:'DoubleAttention'
}