
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    Tensor  xin;
    Tensor  fc1_weight;
    Tensor  fc1_bias;
    Tensor  fc2_weight;
    Tensor  fc2_bias;
    Tensor  fc3_weight;
    Tensor  fc3_bias;
    
    void LoadParameter()
    {
        // input data
        
        xin          ={ -0.49985468,-0.8633351, };
        xin.reshape({1,2});
        
        // {'name': 'Net/Linear[fc1]/weight/43', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
        
        fc1_weight   ={ -0.28409335,0.27795535,-0.26695502,-0.12444893,-0.109930694,0.4120716,-0.31353074,-0.2494081,
                        -0.30830225,-0.5162595,0.25700936,0.62617785,-0.47207773,0.07262053,0.54984796,0.061508942,
                        -0.50814575,-0.06362775,0.3518054,0.63935995,-0.43199095,0.14916849,0.36480272,-0.16002087,
                        -0.54523313,-0.51183665,-0.28351343,0.63494587,0.6246457,0.036428027,-0.33752832,-0.046393987,
                        -0.07671392,-0.4919994,-0.60756874,0.5895185,-0.22857256,0.23247282,0.56752616,0.60955775,
                        -0.08841557,-0.00042585176,-0.6147282,0.21135566,0.17531335,-0.577676,0.64484435,0.12390887,
                        -0.458408,0.5657161,-0.5368261,0.68453723,0.01669403,0.67789656,0.47669145,0.6137972,
                        -0.4604999,0.089943565,-0.13028754,0.38099697,0.0064466135,-0.405497,0.018315166,0.54025704, };
        fc1_weight.reshape({32,2});
        
        // {'name': 'Net/Linear[fc1]/bias/42', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
        
        fc1_bias     ={ 0.3191388,-0.5241661,0.19851662,0.09568591,0.3374156,0.42826408,-0.17881508,0.4536396,
                        -0.08080579,0.05628644,-0.41587138,-0.52329,0.6521526,-0.45658886,0.18798557,0.14534728,
                        0.5678708,-0.28810212,0.55157274,0.051902074,-0.6073111,-0.47662708,0.22179155,-0.34458876,
                        -0.6479991,-0.27971262,0.49808648,0.53409415,-0.3649214,0.63861495,-0.49668458,0.47869724, };
        
        // {'name': 'Net/Linear[fc2]/weight/46', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
        
        fc2_weight   ={ -0.061801102,-0.1453081,-0.027208235,-0.17526935,-0.047168076,-0.15610984,-0.16554673,0.10032047,
                        -0.03680764,-0.11992165,0.0736337,-0.14069952,0.020952504,-0.14155543,0.07171326,0.113247715,
                        0.08984975,0.047386356,-0.08450616,0.1088578,-0.042427674,-0.0885044,0.085083716,0.091397546,
                        0.08398657,-0.022248246,-0.1260439,0.050419766,0.022344595,-0.12532575,-0.14314997,-0.06345081,
                        -0.1662089,-0.016623138,-0.04006452,0.11943104,0.060812,-0.15757598,-0.11559827,-0.10143069,
                        0.055763587,0.034984857,0.092753686,0.020846948,0.05628037,0.16296458,0.038024865,-0.05284374,
                        0.036521338,0.17556278,-0.029730765,-0.023463888,0.029700357,-0.07986078,0.016583731,0.15792416,
                        0.043042067,0.12854576,-0.067761384,-0.011239616,-0.1500398,0.009127554,-0.16279441,0.14173016,
                        0.05047983,-0.04770981,0.038732994,0.12893659,0.01640783,0.022355173,-0.03410504,-0.12022955,
                        -0.16497482,0.028123833,0.00049082114,0.012669174,0.14514858,-0.062881894,0.06833823,-0.07376678,
                        0.1594856,-0.09821141,0.015181547,0.14810461,-0.15969506,0.08113319,0.012936933,-0.019394463,
                        -0.1503624,0.00077305746,-0.17658244,0.025794903,-0.12008206,-0.07614637,0.0784879,-0.16096872,
                        -0.16763034,0.10930462,-0.061838087,-0.02810021,0.017962944,-0.08105844,-0.015169831,0.08916787,
                        0.09503056,-0.1276704,-0.040520526,-0.14765887,0.09522016,0.15799977,0.050070792,-0.040157117,
                        -0.079880476,0.04420155,-0.13790181,-0.04671078,-0.11487945,0.062528074,-0.07509736,0.01558327,
                        0.060379785,-0.01968578,0.11053426,0.012605954,0.05568964,-0.14506848,-0.17167814,-0.1504376,
                        -0.13732849,-0.10036465,-0.003691242,-0.046195894,0.15439235,0.022160728,-0.012331051,0.09989829,
                        0.16093561,-0.15228263,0.024875576,-0.10194315,-0.00929319,0.12998112,0.030179102,0.018960645,
                        -0.09333055,-0.13902487,0.13576132,-0.15194352,0.15340683,-0.035396818,0.13689075,-0.06263384,
                        0.094573416,-0.17519236,-0.11174147,0.039000016,-0.08145498,-0.09335866,-0.17274778,-0.010006716,
                        0.033522084,0.12537347,0.03198444,-0.15444839,-0.06296937,-0.06496401,0.15405864,-0.021105876,
                        -0.058209434,-0.104606815,-0.111831225,0.080914386,0.051192828,-0.15309037,0.06968452,0.13144997,
                        -4.7794525e-05,0.019297313,0.161943,-0.06805768,0.044763345,-0.17216031,-0.08635807,0.0024146351,
                        -0.10982318,0.016922234,-0.056332927,-0.15693374,0.046483062,-0.024673166,-0.17482495,0.060378477,
                        -0.078255825,-0.0067400606,-0.032136675,-0.13666248,0.042829435,-0.057381418,0.03714737,0.13158162,
                        -0.014145767,-0.087742954,-0.033679377,0.06007205,-0.1685889,-0.15734999,0.15615328,-0.056871608,
                        -0.07455859,0.07324785,-0.10070846,0.14019556,0.04743398,0.15321629,0.102630906,0.019057097,
                        0.16156322,0.091092445,0.08021651,-0.07158075,-0.13448983,0.1691319,0.09364694,-0.14598146,
                        0.05516213,0.10394127,0.07419417,0.087862104,-0.16239513,0.081347756,0.08182758,0.061127935,
                        -0.13471782,0.094483346,-0.08254104,-0.15513344,-0.17381105,-0.15900558,0.15751609,0.030663012,
                        -0.12050542,-0.078481585,0.11193672,0.16753791,-0.15803848,-0.15176307,-0.03529828,-0.11221457,
                        -0.040517725,0.15981516,0.056665193,0.07551236,0.10028592,0.10445949,-0.02147955,0.0072969683,
                        -0.08014765,0.08196483,-0.08369574,0.018684836,-0.061216757,-0.12918837,0.13132782,-0.1523374,
                        -0.052062336,0.0018710672,0.012351703,-0.08237157,0.107068926,-0.053227276,0.12301797,0.14056145,
                        -0.17367758,0.0040211044,-0.032541875,-0.044522077,-0.16988075,-0.16002752,-0.11402425,-0.08742921,
                        -0.0969592,-0.17355943,0.0685132,0.13620803,0.10899892,0.15602377,0.066718906,0.08381746,
                        0.13991608,0.11767967,-0.09490391,0.111817256,-0.094467565,-0.04920257,0.11055171,-0.0260893,
                        0.048201665,0.041737728,0.027223112,-0.12450044,0.07239583,0.12132971,0.08597348,-0.004435766,
                        0.086849906,-0.106966026,-0.08194452,-0.066745706,-0.010431261,0.12201689,-0.023916481,0.0024503544,
                        -0.0816235,0.12508447,-0.031530477,0.08757744,-0.15093364,-0.14190212,0.10582433,0.15598275,
                        -0.07184971,-0.12581788,0.08754484,0.08741204,0.17415287,0.13572347,0.071081966,-0.032141585,
                        0.15345971,-0.12268821,-0.04428093,-0.16759738,-0.060802896,0.10318876,-0.14222443,0.06767014,
                        0.13215353,-0.11990391,-0.12360398,0.12850395,-0.021283336,-0.025926402,0.10890493,-0.06545066,
                        0.13003641,-0.15647207,0.021494998,0.17499742,-0.027765563,0.07305819,0.09474217,-0.1207936,
                        0.057123076,-0.07449365,0.0027151632,-0.043488234,0.14833808,-0.0112929735,-0.08842082,0.07021349,
                        0.025472945,0.09691187,0.04840751,0.14616908,0.14540043,0.09465609,-0.0748179,-0.15591757,
                        0.16267982,0.14009903,0.08463315,-0.06377815,0.0922456,0.07855696,0.16906597,0.11511832,
                        -0.07737034,0.17206179,0.112133205,0.11093183,-0.04266182,-0.084303685,0.079181746,-0.09563954,
                        0.004028227,0.10551107,0.064362556,-0.040447675,0.08397498,-0.10977138,0.014658547,0.028349381,
                        -0.09484318,0.13297383,-0.16683272,0.09069437,0.16630276,0.024978962,0.09433809,-0.1041143,
                        -0.1072833,-0.13290733,-0.17256534,0.065472744,0.1747119,0.008479735,-0.1157387,0.12096927,
                        0.13551375,0.13300456,0.14415301,0.10530913,-0.08629352,0.14203201,-0.022594925,-0.010304883,
                        -0.056229353,-0.0096734185,-0.050076146,0.15271878,-0.0884389,-0.041388374,-0.1648611,0.03237116,
                        -0.045362253,-0.16031711,0.13452353,-0.009173261,-0.07475875,-0.083176844,-0.06625923,-0.15260161,
                        0.09354065,-0.15159054,0.025292281,0.008565925,-0.043716818,0.12584695,0.119232,0.016000524,
                        0.067355976,-0.04384724,0.15781417,0.073770896,0.006327443,0.14329611,0.023450254,-0.07188916,
                        -0.112785235,0.07014352,-0.04314124,0.071015246,-0.062974595,-0.14754374,0.11885908,-0.085579656,
                        -0.015692241,0.028829962,0.14235048,-0.050319355,0.14185946,0.03816283,-0.14967123,-0.029593999,
                        -0.10356918,0.05350262,0.06529561,-0.004059479,-0.03096137,-0.057040427,0.119746,-0.14425834,
                        -0.02966685,0.14675589,0.052514657,-0.13966838,0.009553363,0.0844698,-0.14370175,0.1336748,
                        -0.17449969,0.17040795,0.13350256,0.09263426,0.0629032,-0.045908052,0.016099442,0.062423423,
                        -0.12487087,0.10017269,0.030134175,0.049460676,0.05436307,-0.12380409,-0.13830172,-0.16516751,
                        0.02699335,-0.09983933,0.009009479,0.036060043,0.06488537,-0.02829828,0.014057681,0.14877306,
                        0.15366039,-0.16929017,0.17635483,-0.038766522,-0.16876128,-0.03248957,0.14690194,-0.008512546,
                        0.17302243,-0.12737125,-0.025763083,-0.10061778,-0.06053857,-0.105549976,-0.046492737,0.029358841,
                        0.098937325,-0.16207644,-0.023314498,0.043965865,0.021008728,0.09848861,-0.070952445,-0.011150771,
                        -0.0069605308,0.16406997,-0.0020784086,-0.08302668,0.1679134,0.13647023,0.16237526,-0.05262234,
                        -0.17495622,0.08723013,0.16218445,-0.09149874,0.0102879405,0.10072315,-0.047745276,-0.022904092,
                        -0.087321654,-0.117645845,0.0011747802,-0.055908594,0.0674645,-0.15900558,-0.026375119,-0.11880815,
                        -0.008289905,-0.073585615,-0.027509669,-0.14610258,0.115117855,0.007920109,-0.079133175,0.13581575,
                        -0.024594413,0.081435256,0.04455099,0.014177188,-0.04937046,0.10387192,-0.012551774,0.027571898,
                        -0.15909672,0.13322239,-0.13664933,0.15297443,0.09123524,0.004333539,0.10716378,0.034944203,
                        -0.14035408,-0.11522746,0.16057527,0.06682326,-0.1508305,0.047915615,-0.10332363,0.14704312,
                        0.12222655,0.08590375,-0.12342236,-0.13321474,-0.16355486,-0.14361607,-0.0862707,0.0008884134,
                        0.16181983,0.17469008,-0.06626704,-0.054768078,0.058025,-0.09646212,-0.014571219,-0.10960092,
                        0.16649859,-0.015837563,0.034830935,0.12329892,0.02328761,-0.023755545,0.14859122,0.0071681673,
                        0.12512103,0.11231606,-0.09897604,0.042904396,0.14916798,-0.106838994,0.06245208,-0.10237836,
                        0.030911868,0.0022669313,-0.115634784,0.0672697,0.1651897,0.1077429,0.047589418,-0.03716646,
                        -0.052302152,-0.1753577,0.04642294,-0.11225703,0.036860496,-0.047922567,-0.0893635,0.016783422,
                        0.052469537,-0.1035818,0.15341528,0.012017415,-0.032158952,0.114242844,0.1563248,-0.12380255,
                        0.10352876,0.08527173,0.15817282,-0.12092643,0.046092656,0.09897806,0.012350333,-0.03157433,
                        -0.04731222,-0.080087,-0.0008416093,-0.15955622,0.021569682,-0.17603257,0.13050412,0.07677756,
                        0.1052214,-0.03366475,-0.030431647,0.09361565,-0.17404793,-0.043031175,0.053985897,-0.029442018,
                        0.100100115,0.023160936,0.047673035,0.08706964,0.029350284,0.088375494,-0.16244647,0.030336458,
                        0.15795654,-0.117189124,-0.17511393,-0.11169709,-0.13667038,0.08541801,0.1522612,-0.08094258,
                        -0.056058597,0.10732857,0.14951625,0.16458374,0.022537543,0.16726637,-0.121566765,0.1086489,
                        -0.051453672,-0.05652708,0.099098176,0.16181816,-0.0983172,0.06255334,0.10420269,-0.116712205,
                        -0.14565367,-0.08113194,-0.10375142,0.0030383873,-0.021718545,-0.16749106,-0.10939944,0.0060127117,
                        -0.025399819,-0.1353805,-0.05764833,0.10834505,-0.051722843,-0.1680316,-0.007342529,-0.08962384,
                        0.07719218,0.1190766,-0.042738546,0.061900087,-0.115051284,0.028895669,-0.13565953,0.111783326,
                        0.13137116,-0.08860317,-0.070507,-0.060832232,-0.014563801,-0.11396364,-0.11515748,0.078804806,
                        -0.16418673,0.14700699,0.101457834,-0.0022718837,-0.034334403,-0.14132124,-0.106598355,-0.09373971,
                        -0.04304213,-0.05223788,0.12414217,-0.02368754,-0.1391313,-0.03298916,0.10896538,-0.1248389,
                        0.021954441,-0.16851158,-0.10767626,-0.05182678,-0.038968764,0.06085069,0.070644505,0.17244513,
                        0.02376745,0.09351463,0.026124407,-0.059770424,0.086756505,0.13029142,-0.15846819,-0.037816998,
                        0.16175376,-0.16603656,-0.16976704,-0.08467614,-0.009922864,-0.11585184,-0.0023576103,-0.086611286,
                        0.17670797,-0.14291704,0.16618842,-0.13184345,0.13656853,0.055896793,0.15903796,-0.07492559,
                        -0.08115338,0.02991697,0.08745307,0.04243806,0.10600752,0.15613204,-0.11841949,0.11587587,
                        0.009951081,-0.038223315,-0.024611145,0.047049664,-0.07281032,-0.09806836,-0.1685076,0.15028986,
                        0.0018995374,0.12843712,0.041560248,-0.040562253,-0.13411173,-0.08495207,-0.0740094,-0.17086157,
                        0.15322034,0.09186837,-0.13837558,-0.033315293,0.14043869,0.06589434,0.0110505875,-0.030518258,
                        -0.13211975,0.093384095,-0.0338588,-0.075463384,0.14054048,0.024204893,0.0010110608,0.0064090393,
                        0.06757918,-0.062282864,0.13084707,0.06530816,0.091088355,-0.11807806,0.10578912,0.15773702,
                        0.14874832,0.11211201,0.14148532,0.1271123,-0.030493138,-0.14843823,-0.06160571,0.14599325,
                        0.012382133,-0.06912456,-0.12875503,-0.03544925,0.17617336,0.102508195,-0.046206053,-0.109500796,
                        -0.1676755,0.038854633,0.1530308,0.100412466,0.1666514,0.13519605,0.11521338,0.014404486,
                        0.035899483,0.0051570884,-0.07062244,-0.02031105,0.14968577,-0.011351874,0.029519526,-0.02343725,
                        -0.051039666,-0.13897908,0.13829376,0.02937022,-0.15519382,-0.10125896,-0.0065090116,0.05380178,
                        -0.13677558,-0.13302107,-0.0037458853,-0.02758349,-0.012130938,0.013707061,-0.13939188,0.11391054,
                        -0.08762979,-0.13135019,-0.0067161843,0.110540815,0.076856315,-0.08588425,0.12891094,-0.077508114,
                        0.04106852,0.017365387,0.115668945,-0.054444246,0.022037301,0.16896558,-0.021955853,0.02138805,
                        0.056469128,-0.033755768,-0.10891508,0.15482433,-0.13861474,-0.090946555,-0.024514334,0.11978524,
                        -0.17192782,0.044299286,-0.03162767,-0.09174412,0.17217366,-0.08802253,0.134598,-0.17126386,
                        -0.09447016,-0.031888325,0.1438637,0.14829503,0.034127694,-0.10172148,-0.082395546,0.10089462,
                        0.08053496,0.14803979,-0.05108177,0.1166864,-0.1385044,0.02926382,0.08143233,-0.018929372,
                        0.15631863,0.1067139,-0.13014883,-0.0134156365,-0.06368698,-0.08029493,0.14080907,0.090144224,
                        0.13476016,0.06073719,0.058318425,0.07432352,0.011562019,-0.11146044,-0.11927181,-0.008829617,
                        0.14899142,0.07085043,-0.12159018,0.1729738,0.08413912,-0.07582115,0.08751298,-0.15818293,
                        -0.01264475,-0.15147096,-0.12556128,-0.10663134,0.1703688,-0.054245543,-0.09789482,0.09319677,
                        -0.11334927,0.019557128,0.14380896,-0.1300526,0.049878076,-0.16792528,0.045809515,0.035221193,
                        -0.054541394,-0.10298844,0.12370104,-0.120236635,-0.013272443,0.12508397,0.10753368,0.1703892,
                        -0.00048782869,0.12465881,-0.17111273,-0.02595521,-0.036772154,-0.0011689638,-0.12570356,-0.07941949,
                        0.034991514,-0.12719114,-0.093175404,0.11424856,0.15046589,0.066548,-0.042140506,0.0726697,
                        0.061306763,-0.03503343,0.13065305,-0.1383364,0.12920645,0.16083284,0.16318423,0.09292402,
                        -0.13955818,0.16437924,0.027901761,-0.080229856,0.15073986,-0.12550788,0.111695595,0.06930042,
                        -0.07395065,-0.10035957,0.17128958,0.08286888,-0.08296708,-0.035307657,-0.0026495405,0.0684708,
                        -0.10717697,0.018705003,0.046465255,0.039723825,-0.06599225,0.004502021,-0.12890841,-0.013902664,
                        -0.08209713,-0.10490844,0.011765356,0.1550699,-0.06555116,0.055991728,-0.102418445,0.13536954, };
        fc2_weight.reshape({32,32});
        
        // {'name': 'Net/Linear[fc2]/bias/45', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
        
        fc2_bias     ={ -0.067977406,-0.115400955,-0.020779576,0.059536446,0.13065417,0.16199774,-0.048249375,-0.16139811,
                        -0.13286777,-0.15432975,0.15733978,0.15812609,0.01808205,0.056349576,0.0057529183,0.15222615,
                        -0.1299821,0.00876842,-0.12187823,-0.01978706,0.16035634,-0.09014549,0.16824538,-0.03662184,
                        -0.16951966,-0.07418922,0.03979282,-0.06950757,-0.16622455,-0.01224779,-0.105903186,-0.15061252, };
        
        // {'name': 'Net/Linear[fc3]/weight/49', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 9}
        
        fc3_weight   ={ 0.13648091,0.13242714,0.08413084,-0.09080039,-0.0131016215,-0.055015776,0.08576389,0.03837072,
                        0.07630255,0.08704962,0.1382915,-0.08359608,0.1490834,0.0735653,0.13976352,-0.024425216,
                        0.10748053,0.12701823,0.1318855,0.11849626,0.13791455,0.052944344,-0.02787074,-0.13659295,
                        0.06200252,-0.018178126,-0.13631292,-0.17044953,-8.96042e-05,-0.09348048,-0.08530537,-0.06747959,
                        0.013663734,-0.037095547,-0.05748177,0.0722091,0.045503464,0.17296706,0.07643898,-0.05962072,
                        -0.004239172,-0.0023140516,0.17122209,0.033298012,-0.09061277,-0.13145587,0.073897,-0.017869063,
                        -0.15689425,0.14811532,0.11340243,0.17304446,0.029949887,0.16788548,0.046757482,0.14697476,
                        -0.045655806,-0.1077528,-0.08082347,-0.13970102,-0.035592508,-0.11185866,0.06257399,0.07998865,
                        0.107190646,0.06720104,-0.114731625,-0.055028465,0.04052276,0.13801166,0.029275265,0.017454969,
                        0.015775248,-0.08426925,0.12599176,0.035924222,-0.13196771,-0.066160224,0.08822701,0.073253624,
                        0.14137673,-0.14977585,-0.12491639,-0.0028936972,0.058282893,-0.010453135,-0.10780464,-0.1001973,
                        0.17496715,0.071700685,-0.1738932,-0.13433933,0.037388574,0.16393295,0.14109144,-0.12260206, };
        fc3_weight.reshape({3,32});
        
        // {'name': 'Net/Linear[fc3]/bias/48', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 10}
        
        fc3_bias     ={ -0.016351966,0.06499318,-0.030796407, };
        
    }
    