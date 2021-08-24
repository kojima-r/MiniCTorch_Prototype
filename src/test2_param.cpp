
    #include <xtensor/xarray.hpp>
    
    #define fprec float
    typedef xt::xarray<fprec> Tensor;
    
    // input data
        
    Tensor xin ={ 0.66135216,0.2669241, };
    
    // {'name': 'Net/Linear[fc1]/weight/43', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 1}
    
    Tensor fc1_weight ={ -0.6656964,0.42406413,-0.14546975,0.35973623,0.09829985,-0.08658109,0.19612378,0.03488284,
                        0.2582553,-0.27556023,-0.051554278,-0.06365894,0.10249085,-0.002824766,0.6181292,0.22004464,
                        -0.26333097,-0.42706516,-0.11852216,-0.3050039,-0.22659132,0.033857405,0.42152596,0.38433847,
                        -0.69123286,0.43834996,0.19754009,0.67073,0.46673277,-0.6442716,-0.67232305,-0.3410603,
                        0.62091875,-0.117782064,0.30261302,-0.3286006,0.6938259,-0.29917577,0.5302769,0.0083733145,
                        -0.37251619,0.36346337,-0.3753465,0.20796342,-0.20418218,-0.077523224,-0.679804,-0.33713558,
                        0.3837002,-0.17188159,0.70431334,0.5668086,-0.0331093,-0.47198182,0.43060127,0.2194556,
                        -0.45708779,0.4592974,0.4292858,0.62712944,-0.3963755,-0.116394,-0.013697978,0.103278235, };
    
    // {'name': 'Net/Linear[fc1]/bias/42', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [3], 'sorted_id': 2}
    
    Tensor fc1_bias ={ -0.53663623,-0.5017788,0.3846693,-0.16581084,0.34540287,0.04031211,0.23217484,0.15548478,
                        0.25710362,0.3505204,-0.6548601,0.3559232,-0.49718317,-0.5335184,0.04300226,-0.120496064,
                        0.4153008,-0.40951073,-0.6286051,0.51461023,-0.104851075,0.3977173,0.22732392,-0.5302419,
                        0.14205654,0.16984463,-0.47344944,-0.33550256,0.24110618,0.12665878,-0.30078384,-0.21405245, };
    
    // {'name': 'Net/Linear[fc2]/weight/46', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 5}
    
    Tensor fc2_weight ={ 0.16190672,-0.03270745,0.09966522,0.07654487,-0.11426971,-0.15033612,0.16968594,0.009230708,
                        0.12117301,0.03663368,0.056855403,0.13204487,0.16762225,-0.11730566,0.022102207,0.1319209,
                        0.12806323,0.109820485,-0.12793663,-0.12729764,-0.10690953,0.022208165,0.17618713,-0.111665525,
                        0.094200924,-0.0978438,-0.16620822,-0.03757206,0.10185359,0.16411504,-0.109785184,0.038362525,
                        0.15253358,0.11714171,0.11017131,0.12562327,0.111814156,0.04566078,-0.12088346,-0.14844973,
                        -0.08100272,-0.02058492,-0.10835567,0.0646744,0.05471023,-0.040028483,0.067942485,0.05713153,
                        0.10792906,0.11904915,-0.05986591,0.17271537,-0.020436754,-0.00607334,-0.166856,-0.11377634,
                        -0.103273556,-0.07561144,0.12568285,-0.057767272,-0.1321019,0.068014815,0.056598518,0.11449661,
                        -0.091477014,0.03832784,-0.064355664,-0.03971725,-0.14087659,-0.08056791,-0.054139227,0.07559595,
                        0.032295235,0.043666746,0.17644173,0.17229143,0.120570034,0.0056291963,-0.1222926,0.13815269,
                        -0.044205382,-0.014291574,-0.15229294,-0.034918454,-0.11401881,0.16244116,-0.15282717,-0.13779438,
                        -0.0060026804,-0.09558589,0.06325277,-0.06804588,-0.083029814,0.010012216,0.12796828,-0.12434892,
                        0.08301544,0.11356866,0.17292246,-0.12373147,0.04281353,-0.13071343,0.15092315,-0.0685728,
                        0.106481925,0.005262898,-0.013767837,-0.0056405547,0.03005106,0.08333184,0.028351426,0.0539266,
                        -0.159015,0.12878844,0.15412523,0.14612463,0.13067733,-0.12756762,-0.06553112,0.15586849,
                        -0.13464348,0.16037557,-0.13903417,-0.12452301,0.086422235,-0.12701212,-0.040499665,0.1286036,
                        0.14002284,0.16718757,-0.035871077,-0.1374016,0.17405626,-0.03765806,-0.07273214,0.04309353,
                        -0.12362427,0.11619182,0.110791735,-0.14026898,-0.1451995,-0.0154850045,0.07425071,-0.0051198304,
                        -0.0896395,0.0040415456,-0.16615811,-0.124938875,-0.11767836,0.14558096,0.15582968,-0.06004516,
                        0.007928496,0.078838505,0.02118092,-0.08852138,0.1019539,0.108670674,-0.010245183,-0.021761661,
                        0.16064496,0.15452561,-0.10022853,0.17295612,0.043747373,-0.11742584,0.09676977,-0.13197698,
                        0.16332905,-0.113646045,0.049997393,0.05382854,0.04204774,0.14662252,-0.07342411,-0.074629866,
                        -0.15318963,-0.007254863,-0.08374275,0.0070833047,-0.036238957,0.05865339,0.15292533,0.05216294,
                        -0.05694688,-0.08674365,-0.15417632,0.14874819,-0.0334886,0.03568327,0.006539168,-0.11265397,
                        0.09955735,0.09906897,0.008902468,-0.10046024,-0.07511359,0.002434402,-0.1009991,-0.09863401,
                        -0.026369259,-0.050902918,0.04339528,-0.04646496,-0.023434069,-0.067472406,-0.16154924,0.15417172,
                        0.033725865,0.15807806,-0.02848952,-0.044426255,-0.16618885,-0.067106314,0.1159252,-0.17314315,
                        -0.061830375,0.10770284,0.10188442,0.06527396,-0.034992084,-0.013384764,-0.0335711,-0.06516703,
                        0.07377325,-0.06021065,-0.09930971,0.040103633,-0.12777525,-0.03772676,0.0903353,0.06606666,
                        0.15049674,-0.12057001,0.0015369902,-0.052202642,-0.09887417,-0.09944635,0.044536427,0.018495196,
                        -0.09023539,-0.12460273,-0.14646447,0.17325258,-0.030130781,-0.040877007,-0.06500397,-0.08841766,
                        -0.16166444,-0.103778355,0.10805095,0.038745176,-0.062359508,-0.06680611,0.1129274,0.12742853,
                        0.17031273,0.051566754,0.08540279,-0.014203698,-0.103349954,-0.1731849,0.10767367,-0.02569394,
                        0.07262798,0.008701259,-0.1649894,-0.086725734,0.049732987,-0.03577968,-0.17070661,0.061988425,
                        0.0067796996,-0.056812603,0.071412735,0.037321012,-0.06519578,0.15880702,0.08868525,0.022196554,
                        -0.033950213,-0.1497369,0.1364853,-0.04561549,-0.11423762,0.11334071,0.06278243,0.0437467,
                        -0.035051364,-0.055166747,0.029998735,0.13679847,0.025315145,-0.083588496,0.17560787,0.0037807408,
                        0.13974509,0.043083984,0.15845492,0.10199552,-0.09012555,0.124177724,0.0011372273,-0.14190172,
                        -0.037042845,-0.15956078,-0.091588385,-0.12523495,0.1209067,-0.037138686,-0.030542936,0.08471411,
                        0.07194292,-0.11666694,0.065148346,0.07509873,-0.14055681,0.1536962,-0.10756525,-0.06912867,
                        -0.14560843,0.16157,0.03366138,0.14885212,0.011543663,0.096085854,-0.05644224,-0.14102814,
                        0.028377537,-0.015764691,-0.01784898,-0.08681766,0.0036595897,0.050767396,0.06377183,-0.14240499,
                        0.08540439,0.018178463,0.121926,0.124930546,-0.13921425,0.16976221,-0.17385118,0.10162288,
                        0.012442276,0.08365228,-0.09559489,0.10627751,-0.0874749,-0.1562522,0.05922606,0.0967604,
                        0.17523164,-0.018475408,0.1700554,0.11356803,-0.087674886,-0.13635571,0.09914763,-0.096851684,
                        0.08108655,-0.15139809,0.12897095,0.05298312,-0.020322746,0.08107475,-0.12585557,-0.03864461,
                        0.0017330152,-0.1021463,-0.045170356,-0.07623581,0.16847458,0.1501781,0.15246643,0.14448601,
                        -0.16930026,0.05913222,0.06515789,-0.12966193,0.071654804,-0.09592199,0.028110452,0.058553625,
                        0.15253973,-0.09273965,-0.13908346,0.1635641,0.074205175,0.08592183,0.06363335,-0.06257201,
                        0.10633976,-0.15433213,0.024601031,0.079910085,0.12078174,-0.088277124,-0.11948153,0.06626113,
                        -0.14038971,-0.14691259,-0.15897852,-0.08572515,-0.033742744,-0.104685985,-0.07474581,-0.15653345,
                        0.108654656,-0.040469468,-0.11153645,0.046024695,-0.13217734,0.12374196,0.101649344,-0.1737205,
                        -0.13876018,-0.1165624,-0.095694736,0.022072662,-0.17480683,-0.13105638,0.08073127,-0.057896577,
                        -0.070343405,-0.14624403,0.07164962,0.012483285,-0.06832952,-0.10825519,0.037595514,0.04809948,
                        0.07846428,0.052534506,-0.02696646,-0.13684437,0.11253752,-0.0015153477,-0.117483534,0.15245044,
                        0.09856146,-0.138934,-0.0882853,-0.06015788,-0.10205022,0.082552396,-0.12611148,0.16429077,
                        -0.07306788,0.10434693,0.0060201082,-0.07774481,0.11804843,-0.13488656,-0.09352936,0.021188527,
                        0.14023247,-0.07574092,-0.107644774,-0.11285704,-0.07793679,-0.061064817,-0.041175045,-0.10056577,
                        0.055250704,0.0014356902,-0.11550028,-0.10095091,0.037432827,-0.0025253126,0.1251133,-0.026798652,
                        -0.1432071,-0.13074625,-0.051888376,-0.039151978,0.02017422,-0.03964579,0.065226376,-0.05879306,
                        0.12974595,0.040603388,-0.054347474,0.15754353,0.045130022,-0.12013567,-0.09838393,0.11084164,
                        0.08073762,0.13802621,-0.01357923,0.12364125,0.08788956,0.054669455,-0.04091424,0.17041071,
                        0.03577319,-0.04561081,-0.002513343,0.17378005,0.11873222,-0.013100252,0.17330578,0.07762891,
                        -0.09411892,-0.16087899,0.102731764,0.16578697,-0.1029945,0.07260623,0.007525615,-0.037797946,
                        -0.048067424,0.049179554,0.038971525,0.11939652,0.06471339,0.08044352,0.1559448,0.05504566,
                        0.026241176,-0.06435811,0.12964615,0.026106622,0.11394468,0.1707038,-0.03953663,0.12446862,
                        -0.035899904,0.15952854,0.1391401,-0.06336307,0.0063398974,-0.07719977,-0.07149684,-0.035791524,
                        -0.0012052102,-0.061623137,-0.1668769,-0.106919385,-0.015106273,-0.056432478,0.13130182,0.14369416,
                        0.11261781,0.0782381,-0.1401421,0.16701236,-0.15577741,-0.05811654,-0.12441817,0.06356891,
                        -0.14709863,-0.087149926,-0.11229058,-0.10866212,-0.082522646,0.17377473,-0.014438794,0.05388656,
                        -0.17548339,0.15717481,-0.038494233,-0.1622357,0.12182732,-0.04779933,0.16053656,-0.12007241,
                        -0.16412523,-0.13280416,0.07657583,-0.01982495,-0.16376933,0.03277739,-0.011268065,0.063864045,
                        -0.08992276,-0.12619886,0.12048808,-0.08506507,0.12435088,0.14333157,-0.12362575,-0.08150117,
                        0.021280492,-0.16667674,0.054289483,-0.07365613,-0.07837885,-0.13666019,0.0826994,-0.11265043,
                        -0.14129794,-0.046487954,-0.09841192,-0.17186193,0.071243554,-0.06725748,0.12382598,0.12443615,
                        -0.11274298,-0.103055954,-0.06274812,0.10354783,-0.006085183,-0.063152544,0.10950975,0.16764551,
                        0.13069059,-0.13518211,-0.036716416,-0.03390792,0.15086171,-0.09143038,-0.10131607,0.0027658027,
                        -0.041065,0.066913076,0.021487538,-0.10554508,0.12854296,0.11003971,0.03871169,-0.16000488,
                        0.08307076,0.117461495,0.04921481,-0.068331845,0.011855845,-0.06791043,0.16642715,0.1706984,
                        -0.020600557,-0.11893491,0.10398934,0.09022367,0.093705125,-0.13301612,0.059942555,-0.075439505,
                        -0.15537834,-0.12343907,0.0749368,0.06618142,-0.15515104,0.10024594,0.13545337,0.13419615,
                        0.09981567,-0.027750222,0.118452005,0.13839489,0.09215041,-0.12714669,-0.061696324,0.14004576,
                        0.13477036,0.11279746,-0.073885635,-0.03501031,-0.13170032,-0.10074521,0.13569976,-0.14509457,
                        -0.043685988,0.037524182,-0.10606819,0.041206762,-0.02841764,0.07069719,-0.13565554,0.024395902,
                        -0.16117178,0.10967092,0.17171559,-0.06309554,0.026140003,-0.091077186,-0.113081425,-0.089529596,
                        -0.08421587,0.033556096,-0.020263404,0.012394925,-0.13546947,-0.111062065,-0.1530392,-0.12705715,
                        0.1480358,0.065676905,-0.16859175,-0.029504774,0.014187367,0.09443846,0.16395253,0.042687275,
                        0.12452765,-0.06290495,-0.0057760146,-0.10671415,0.040724117,0.16317472,0.11101201,0.17590718,
                        -0.0796536,-0.022118729,0.036464337,0.14212362,-0.1625004,-0.08681699,0.025643323,-0.102177575,
                        0.06603682,-0.012051743,-0.045807786,0.09480683,0.1263373,0.17194867,0.080051765,-0.0048343698,
                        0.14453381,0.14068645,-0.08859632,0.08052136,0.020714207,-0.07639795,-0.042098694,-0.041742764,
                        -0.12370365,-0.053944424,0.05948665,0.12977487,0.013428006,-0.17119083,0.093623154,-0.053489957,
                        0.15787062,-0.17081578,-0.04135986,0.1370943,0.07647375,-0.08819533,-0.15307221,-0.0018986734,
                        0.06829079,-0.1467595,-0.14792734,-0.0022253324,-0.03625472,0.11362167,-0.17128502,-0.028117215,
                        -0.13698936,-0.025747383,-0.09031924,0.055635273,-0.14788345,-0.13906336,-0.10031915,-0.15482758,
                        0.11159341,-0.025298666,-0.09821474,0.123872474,-0.04010652,0.091427214,-0.05479891,-0.14790009,
                        0.038825,0.11007731,-0.17236291,0.018401798,0.035233565,0.020289619,-0.0043744845,-0.054950174,
                        -0.13723809,0.028833207,-0.1187259,-0.14606132,-0.05675467,-0.12101359,0.054182112,0.1259183,
                        0.13738549,-0.10957006,0.096378,0.12561986,0.0019830302,0.074580066,0.100027554,0.13526028,
                        0.094588466,0.07545626,-0.06709544,-0.09855971,-0.083824836,0.14544892,-0.042816456,0.0063227224,
                        -0.16485931,-0.007852379,-0.0498687,-0.11406309,0.13690701,-0.14312045,-0.025270155,-0.0125341145,
                        -0.16003662,0.10688166,-0.05624819,0.079416566,0.09866242,0.15587965,0.04372563,0.13271259,
                        0.1632616,0.03503621,-0.03601897,-0.04840146,0.028639352,0.063616835,-0.16676794,0.13245103,
                        0.027876474,-0.16467279,0.05662313,0.1121849,0.06605389,0.1442981,0.0036545952,-0.14665602,
                        -0.0004010062,0.12849015,-0.08502688,0.07540071,0.04074953,-0.02868211,0.0012563133,0.07520972,
                        -0.12096971,0.111613385,0.11967802,0.09919739,0.17664818,-0.0077192164,-0.12482889,0.008072048,
                        -0.051287025,0.062404245,0.14953364,-0.096755825,0.007402609,0.17193942,-0.14112033,-0.03050515,
                        0.17166062,0.12352046,0.051307738,-0.07624398,-0.016478911,-0.15852,-0.16642968,-0.16562822,
                        0.1697891,0.13981801,0.07012222,-0.044401724,0.15220329,-0.046694387,0.122723505,-0.0057534873,
                        -0.0831575,-0.049337797,-0.03774678,-0.16181013,0.05286325,-0.053860363,0.08121771,0.11271011,
                        -0.028105667,0.08573314,0.035319228,0.11871777,-0.11400482,-0.015520197,-0.058907278,0.15489742,
                        -0.15867342,-0.09936145,0.019956322,0.15693098,0.119608894,-0.16542022,-0.12428549,0.17140432,
                        0.025762683,-0.1514104,-0.05569356,0.07728421,-0.036965314,-0.13578723,-0.10914474,-0.09157648,
                        -0.0069243056,0.06927728,0.08487553,-0.08587663,-0.029660696,-0.1364868,0.1336862,0.17641503,
                        0.17182226,0.07224423,-0.03769534,-0.004406769,0.06377819,-0.15323164,0.0109737115,-0.1436938,
                        -0.17674616,-0.005908503,-0.056277063,-0.11198101,0.029305018,0.007201168,0.026976217,-0.09893975,
                        -0.17315073,0.0038688488,-0.07211684,-0.059597835,0.13014051,0.028870843,0.080736585,-0.049404155,
                        -0.14026989,0.035248674,-0.11706978,0.021903211,-0.1379605,0.13327041,-0.120422736,0.15347011,
                        0.12487563,0.13363305,0.1060736,0.040916685,-0.008615827,-0.0028217526,-0.070307896,-0.16249251,
                        0.16112742,0.009159227,0.08314146,0.0064882124,0.05080067,-0.10891755,-0.13742013,0.13481739,
                        0.06703656,0.08728579,-0.17546073,0.011951518,-0.16488911,-0.020229792,-0.0093451785,-0.12436595,
                        -0.17504874,0.0011634848,0.15897515,0.15931553,-0.10762543,-0.031402856,-0.14782834,-0.13805443,
                        -0.04846976,0.17162521,-0.046118386,0.010227965,0.16630526,0.03150289,0.14260544,0.117466554,
                        0.16576543,0.030315448,-0.14295618,-0.15207449,-0.09133658,0.14590724,0.10165232,0.051738355,
                        -0.13462059,-0.13019685,0.14422314,-0.018350083,-0.17601138,-0.021853479,-0.0424396,0.0026665046,
                        0.073982574,0.06802134,-0.13526575,-0.0019375328,-0.08760568,0.11957676,0.08753677,0.097029485,
                        0.16071537,0.08683315,-0.17665152,0.044606052,-0.1213016,0.023606913,0.0018895274,0.17533086,
                        -0.12077782,-0.17319895,-0.047390148,0.02848657,0.12949672,0.14333083,0.1361439,0.05580365, };
    
    // {'name': 'Net/Linear[fc2]/bias/45', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [7], 'sorted_id': 6}
    
    Tensor fc2_bias ={ -0.050519317,-0.037378184,-0.096674375,0.034568783,-0.122937694,0.060776934,0.11416426,0.1500069,
                        -0.17424208,0.14503993,-0.11104982,-0.014433799,0.16137487,0.09579618,-0.056074064,0.10435627,
                        -0.17193949,0.13820057,0.17160036,-0.008027731,0.02909376,-0.04148057,-0.11190899,-0.07995834,
                        -0.011739646,-0.14462356,-0.14287344,0.019400742,-0.1150241,-0.09615698,-0.0015287504,-0.093161836, };
    
    // {'name': 'Net/Linear[fc3]/weight/49', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 9}
    
    Tensor fc3_weight ={ 0.16363314,-0.063816294,-0.09829317,-0.047094528,-0.1442274,0.09106794,0.15188573,-0.10194913,
                        -0.06571568,0.16497998,-0.12787923,0.047878124,-0.14403148,-0.1573393,0.00026748498,-0.1008931,
                        0.09573972,-0.054678034,0.0017417396,-0.09229129,0.07264372,0.16378765,-0.026574789,-0.10910843,
                        -0.17241451,0.1693986,0.07450726,0.073006324,0.030097,0.08870388,0.0627752,0.08631577,
                        0.11055891,0.025066944,0.1522645,0.10992332,-0.0054114657,-0.078956805,0.011718636,0.13767573,
                        0.11198424,-0.07163423,-0.03690593,0.07018748,0.0666061,0.168334,0.08584046,0.056318577,
                        0.040856395,-0.16613449,-0.06512765,0.018709386,-0.122976914,-0.15488602,0.119285904,0.060122795,
                        -0.14265275,0.010492058,0.10410781,0.10183661,0.161434,-0.06989804,0.07871366,-0.14339812,
                        -0.021805326,-0.11242918,-0.033661064,0.10063607,0.08812758,-0.092987135,-0.04383504,-0.035365105,
                        -0.06725059,-0.09683729,0.12980683,0.12219324,0.14757805,-0.097070135,0.12945114,-0.053604193,
                        -0.0415322,-0.047594305,-0.008595766,-0.07741025,0.024175601,-0.11393943,0.07270613,-0.14824897,
                        -0.06408567,-0.12347069,0.1192481,0.12145413,-0.14060038,-0.096462,0.15821233,0.05151114, };
    
    // {'name': 'Net/Linear[fc3]/bias/48', 'op': 'prim::GetAttr', 'in': [], 'shape': [], 'out': [11], 'sorted_id': 10}
    
    Tensor fc3_bias ={ 0.1217241,-0.13552706,0.07978365, };
    