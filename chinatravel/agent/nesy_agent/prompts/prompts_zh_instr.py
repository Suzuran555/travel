# -*- coding: utf-8 -*-
"""Chinese-INSTRUCTION variant of the hardened English prompt stack (H1 probe).

Same rules, few-shots, checklists and DSL spec as prompts_en.py /
nl2sl_hybrid_en.py, with the instruction/explanation text rendered in natural
Chinese. Everything the model must COPY or EMIT stays unchanged and English:
python code shapes, function names, value vocabularies, few-shot examples,
POI handling and the json output format.

Activated via PENGUINS_PROMPT_LANG=zh in nl2sl_hybrid_en.py; the deterministic
normalizers / verifiers there are language-agnostic and run identically.
"""

# The few-shot examples are shared verbatim with the English stack: only the
# surrounding instructions differ between the two variants.
from chinatravel.agent.nesy_agent.prompts.prompts_en import (
    nl2sl_example,
    nl2sl_example_1,
    nl2sl_example_2,
    nl2sl_example_3,
    nl2sl_example_4,
)

# ---------------------------------------------------------------------------
# Step 1: NL -> hard_logic (mirrors prompts_en.nl2sl_prompt + rules (a)-(e))
# ---------------------------------------------------------------------------

nl2sl_prompt = """
你需要从自然语言查询中抽取 start_city、target_city、days、people_number，并把查询中的需求转换为 hard_logic 约束。
说明文字是中文，但输出的变量名、取值和格式必须保持英文/原样，与下方示例完全一致。
可用的 hard_logic 变量如下：
(1) days: 必须等于用户想旅行的天数。
"days==n" 表示用户想旅行 n 天。
(2) people_number: 必须等于出行人数。
"people_number==n" 表示 n 人出行。
(3) cost: 必须小于等于用户给出的预算。
"cost<=n" 表示整个行程的花费不超过 n。
(4) tickets: 用户需要购买的票数（整数）。
"tickets==n" 表示用户需要购买 n 张票。
(5) rooms: 用户需要预订的房间数（整数）。
"rooms==n" 表示用户想预订 n 间房。
(6) room_type: 每间房的床数。
"room_type==n" 表示用户想要每间房有 n 张床。
(7) hotel_feature: 用户想要的酒店特色集合，取值必须在 ["Kids' Club", 'Air purifier', 'Mountain View Room', 'Private Hot Spring Room', 'Courtyard house', 'hot spring', 'Lakeside Residence', 'e-sports hotel', 'Hot spring bathing', 'Executive Lounge', 'Charging station', 'Designer hotel', 'homestay', 'Lake View Room', 'Stunning Night Views', 'Luggage Storage', 'Chinese-style courtyard', 'Billiards Room', 'Private Pool', 'Fishing', 'Charming sea view', 'Garden Architecture', 'Old Western-style house', "Children's Pool", 'Historic Residence', 'Mahjong and Card Game Room', 'Smart Room Control', "Couple's Room", 'small and beautiful', 'Tea Room', 'Family-themed room', 'Multifunction Hall', 'Laundry room', 'inn', 'Self-operated family room', 'Parking lot', 'Recommended by the Boss', 'River view room', 'Sunbathing area', 'Self-operated entertainment room', 'Kitchen', 'Air conditioning', 'Instagrammable pool', 'Villa', 'Free parking', 'Laundry service', 'Great view from the window', 'Serviced Apartment', 'Conference Hall', 'Family Room', '24-hour front desk', 'Business Center', 'Early Park Entry', 'Farm stay', 'Smart toilet', 'Gourmet Hotel', 'Spa', 'Photogenic', 'Ocean View Room', 'Swimming Pool', 'Media Room', 'Butler Service', 'Airport shuttle service', 'Sauna', 'Robot Service', "Children's Playground", 'Fitness Room', 'Washing machine', 'Self-operated Comfort Sleep Room', 'Pet-friendly', 'e-sports room', 'Excellent location', 'Suite'] 之内。
"{'A'}<=hotel_feature" 表示用户想订的酒店具有特色 A。
(8) hotel_price: 必须小于等于用户给出的酒店价格（每晚均价）。
"hotel_price<=n" 表示酒店价格不超过 n。
(9) intercity_transport: 城际交通方式集合，取值必须在 ['train','airplane'] 之内。
"intercity_transport=={'train'}" 表示用户想乘火车前往目的地。
(10) transport_type: 市内交通方式集合，取值必须在 ['metro','taxi','walk'] 之内。
"transport_type<={'A'}" 表示用户在市内乘坐交通方式 A。
(11) spot_type: 用户想游览的景点类型集合，取值必须在 ['Museum/Memorial Hall', 'Art museum', 'Red tourism sites', 'natural scenery', 'Cultural Landscape', 'University campus', 'historical site', 'Amusement Park/Sports Entertainment', 'Garden', 'Other', 'Cultural Tourism Area', 'park', 'commercial district'] 之内。
"{'A', 'B'}<=spot_type" 表示用户想游览类型为 A 和 B 的景点。
(12) attraction_names: 用户想去的景点名称集合。
"{'A', 'B'}<=attraction_names" 表示用户想去景点 A 和 B。
(13) restaurant_names: 用户想去的餐厅名称集合。
"{'A', 'B'}<=restaurant_names" 表示用户想去餐厅 A 和 B。
(14) hotel_names: 用户想入住的酒店名称集合。
"{'A'}<=hotel_names" 表示用户想入住酒店 A。
(15) food_type: 用户想品尝的菜系集合，取值必须在 ['Yunnan cuisine', 'Tibetan cuisine', 'Northeastern Chinese cuisine', 'Barbecue', 'Asian cuisine', 'Cantonese cuisine', 'Northwestern Chinese cuisine', 'Fujian cuisine', 'Hakka cuisine', 'Fast food and casual dining', 'Sichuan cuisine', 'Taiwanese cuisine', 'Other', 'Halal cuisine', 'Snacks', 'Western cuisine', 'Vegetarian cuisine', 'Japanese cuisine', 'Jiangsu-Zhejiang cuisine', 'Hubei cuisine', 'Southeast Asian cuisine', 'Hunan cuisine', 'Beijing cuisine', 'Korean cuisine', 'Seafood', 'Middle Eastern cuisine', 'fusion cuisine', 'Teahouse', 'Bar/Pub', 'Creative Cuisine', 'buffet', 'coffee shop', 'Shanghai cuisine', 'Huizhou cuisine', 'Latin American cuisine', 'Shandong Cuisine', 'Xinjiang cuisine', 'Farmhouse cuisine', 'Hainan cuisine', 'Hot pot', 'Bakery and Desserts', 'Other Chinese Cuisine'] 之内。
"{'A', 'B'}<=food_type" 表示用户想品尝菜系 A 和 B。
(16) food_price: 必须小于等于用户给出的餐饮价格（每餐人均）。
"food_price<=n" 表示每餐价格不超过 n。
你的回答必须是合法的 json 格式。注意 hard_logic 的格式并参考下方示例。
(17) taxi_cars: 用户需要乘坐的出租车数量（整数），可由 `(people_number+3)//4` 计算。
(18) activity_start_time: 活动的开始时间。
(19) activity_end_time: 活动的结束时间。
(20) activity_time: 活动的时长。
如果行程只有一天，忽略 rooms 和 room_type；其他不需要的约束同样忽略。
如果你发现某些约束不在上述范围内，也可以把它们加入 hard_logic。

规则（必须严格遵守）：
(a) 永远输出基础约束：'days==N'、'people_number==N'、'tickets==N'（N = people_number）以及 'taxi_cars==M'（M = (people_number+3)//4，取整数）。只有当查询明确给出预算 B 时才输出 'cost<=B'。
(b) 除非查询明确提到房间数、床数或床型/房型，否则绝不输出 rooms、room_type 或 room_count（任何拼写或形式都不行）。只出现酒店名、酒店特色或酒店价格并不意味着需要 rooms/room_type。
(c) 每个景点/餐厅/酒店名称必须从查询中逐字复制为一个完整字符串：保留括号、'·'、分店后缀和空格；绝不把一个名称拆成两个，绝不缩写或翻译。
(d) 查询中的每一句需求必须恰好映射为一条 hard_logic；除 (a) 的基础约束外，每条 hard_logic 都必须在查询中有明确出处。不得凭空发明约束。作答前重读查询并双向自查：没有遗漏需求，也没有多造约束。
(e) 如果查询说行程只需满足若干编号条件中的至少/任意一个（'at least one of the following'、'any one of'、'either of the following'、'满足以下条件中的至少一个'），必须输出一条用 ' or ' 连接各分支表达式的 hard_logic 字符串（例如 "hotel_price<=3300 or ({'Xidan Commercial Street'}&attraction_names)==set()"）。绝不能只保留一个分支作为无条件约束，也绝不能把分支拆成多条独立项（那等价于 AND）。
"""


class NL2SL_INSTRUCTION:
    def __init__(self):
        pass

    @classmethod
    def format(cls, nature_language):
        return (
            nl2sl_prompt
            + nl2sl_example
            + nl2sl_example_1
            + nl2sl_example_2
            + nl2sl_example_3
            + nl2sl_example_4
            + "\nExamples End."
            + "\nnature_language: "
            + nature_language
            + "\nlogical_constraints: "
            + nature_language
            + "\n"
        )


# ---------------------------------------------------------------------------
# Step 2: hard_logic -> hard_logic_py (mirrors nl2sl_hybrid_en.sl_trans_prompt)
# ---------------------------------------------------------------------------

_SL_TRANS_HEAD = """
下面提供了一组函数。请把自然语言约束翻译成 python 代码，并以 json list 格式输出。
（说明文字是中文，但输出的代码、函数名和字符串取值必须保持英文/原样。）
variables:
(1) plan: a dict of the generated plan with information of the specific plan.

functions:"""

_SL_TRANS_BODY = """
你需要按如下格式回答：
[
    "python code block 1",
    "python code block 2",
    ...
]

并非所有约束都需要翻译成 python 代码；无法翻译成合法 python 代码的约束请忽略。
!!! 代码中只能直接使用 `plan` 变量，其余变量必须在代码中用上面提供的函数定义。!!! 注意函数返回值的类型！！！
通常，对"存在性"约束，可以在代码开头设 result=False，条件满足时置 result=True；对"全部都要满足"的约束，在代码开头设 result=True，条件不满足时置 result=False。

### 硬性规则（违反这些规则的代码会崩溃或被拒绝）
1. 执行器只暴露内建函数 `set`。len、bool、any、all、sum、str、int、float、map、sorted、abs、max、min 都未定义，使用会抛 NameError。非空判断写成 result=(A&B)，判空/否定写成 result=not(A&B)，子集判断用 A<=B，计数用循环内显式递增的计数变量。
2. 输出列表中的每个字符串都会在全新的命名空间中独立执行：它必须完全自包含并给 `result` 赋值。绝不能引用另一个字符串中定义的变量。"至少满足其一 / either ... or"类需求必须写成一条约束：在同一代码块内计算所有子条件并用 `or` 连接。
3. 每个 POI 名称必须从请求中逐字复制为一个字符串：完整的原始子串，包括括号、'·'、分店后缀和空格。绝不在内部分隔符处拆分一个名称，绝不翻译、缩写或归一化。只有当明确的分隔符（逗号 / 'and'）分隔的是明显不同的场所时才拆分列表。
4. 永远输出基础约束，形式与示例完全一致：days、people、tickets 约束（attraction/airplane/train 票数和 metro 票数 == 人数）以及 taxi_cars 约束。除非请求明确提到房间、床、床型或酒店房型要求，绝不输出 room_count/room_type 或任何住宿房间约束：任何形式的 `room_count(activity)!=N` 或 `room_type(activity)!=N` 检查都不允许，无论是单独成条还是嵌在别的循环里。
5. 作答前自查：请求中的每个需求子句恰好映射为一条约束；除基础约束外，每条约束都能在请求中找到出处；没有出现被禁止的内建函数。

### 标准写法（严格照抄这些代码形状）
- 必须游览/就餐/入住 X：按正确的 activity 类型收集名称集合，然后 result=({'X'}<=name_set)
- "X、Y 里去一个 / any of X, Y"：result=({'X','Y'}&name_set)（一条约束；用交集 &，不要用 <=）
- "不想去 / 避开 X"：result=not({'X'}&name_set)
- 市内交通方式偏好（不打车 / 不走路 / 只坐或尽量坐地铁 等）：一律把被禁止的方式写成如下黑名单形式（"只坐地铁"禁止 walk 和 taxi）：
"inner_city_transportation_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='transportation': inner_city_transportation_set.add(activity_position(activity))\nresult=not({'walk', 'taxi'}&inner_city_transportation_set)"
绝不要用 innercity_transport_type 或 activity_transports 表达方式偏好，也绝不要把黑名单改写成白名单。
- "在 A 到 B 之间（或 from A to B）游览 X"：该活动必须覆盖整个时间窗，即开始不晚于 A 且结束不早于 B。只按 activity_position 匹配：
"result=False\nfor activity in allactivities(plan):\n  if activity_position(activity)=='X':\n    if activity_start_time(activity)<='A' and activity_end_time(activity)>='B': result=True"
不要写成 activity_start_time>='A' and activity_end_time<='B'。
- 预算上限是分范围的——把预算名词映射到它自己的聚合，绝不映射到 total_cost 写法：餐饮/用餐预算 -> restaurant_cost 对 ['breakfast','lunch','dinner'] 累加 activity_cost；住宿/酒店预算 -> accommodation_cost 对 'accommodation' 累加 activity_cost；景点/门票预算 -> attraction_cost 对 'attraction' 累加 activity_cost；城际/跨城交通预算 -> inter_city_transportation_cost 对 ['airplane','train'] 累加 activity_cost；市内/本地交通预算 -> inner_city_transportation_cost 对所有活动（不加任何 activity_type 过滤）累加 innercity_transport_cost(activity_transports(activity))。每条以 result=(累加变量<=上限) 结束。只有明确的"总预算/总花费"才用示例中的 total_cost 写法；把分范围预算写成 total_cost 会让查询无解。
- "只去/只参观免费景点"："attraction_cost=0\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction': attraction_cost+=activity_cost(activity)\nresult=attraction_cost<=0"
- 有方向的城际交通——"坐X去(目的地)/坐Y返回"（含否定"不想坐X去"）只约束第一个/最后一个活动，绝不约束全局交通方式集合：
"result=False\nintercity_transport_go=''\nintercity_transport_back=''\nif allactivities(plan)[0]['type'] == \\"train\\" and intercity_transport_origin(allactivities(plan)[0])==start_city(plan) and allactivities(plan)[-1]['type'] == \\"airplane\\" and intercity_transport_origin(allactivities(plan)[-1])==target_city(plan):\n  result=True"
（否定用 !=；未提到的一程省略对应子句）。出现"去程/返程/回来"等方向词时，全局 intercity_transport_set 约束是错误的且常自相矛盾。
- "到达 X 不晚于 T" -> 对开始时间做存在性检查："result=False\nfor activity in allactivities(plan):\n  if activity_position(activity)=='X':\n    if activity_start_time(activity)<='T':\n      result=True"。"离开 X 不早于 T" -> 同形状但用 activity_end_time(activity)>='T'。绝不颠倒 start/end，绝不写成空洞的全称（result=True ...）形式。
- "两地距离超过 D 公里就打车"："result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity)) != 'taxi' and innercity_transport_distance(activity_transports(activity))>D:\n    result=False\n    break"
- "住宿在 X 附近 D 公里以内"："result=False\naccommodation_position=''\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation': accommodation_position=activity_position(activity)\nresult=(poi_distance(target_city(plan), 'X', accommodation_position)<=D)"

### 析取（'must meet at least/any one of the following'、'either of the following'、"至少满足以下条件之一"）
这类请求列出若干编号分支，但规划只需满足其中一条。必须翻译成恰好一条自包含约束：在同一代码块内计算每个分支条件，并用 `or` 连接。
NL: '... must meet at least one of the following: 1. Do not wish to visit Xidan Commercial Street and Dingling Mausoleum; 2. Accommodation budget is 3300.0.'
正确（一条约束）：
"attraction_name_set=set()\nhotel_cost=0\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\n  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\nbranch_1=not({'Xidan Commercial Street', 'Dingling Mausoleum'}&attraction_name_set)\nbranch_2=(hotel_cost<=3300.0)\nresult=(branch_1 or branch_2)"
错误——折叠（禁止）：只输出一个分支，例如只写 "result=(hotel_cost<=3300.0)"，会把"至少满足其一"变成无条件硬性要求。
错误——拆分（禁止）：把 branch_1 和 branch_2 输出为两条列表项意味着两者都必须满足（AND 语义）。

### 注意!!!
如果自然语言约束中出现上面函数未定义的伪代码，必须用上面提供的函数把它改写成 python 代码块。通常，对景点和餐厅，只要所需的那一项出现即满足要求；但住宿通常整个行程都住同一家酒店，所以需要检查规划中所有住宿活动。
###

如果发现自然语言约束本身有错误，需要在代码块中修正。例如出现 {'natural landscape'} <= spot_type 时，应改用上面提供的 'natural scenery'。

Example:
nature_language:
days==2
people_number==3
cost<=3000
tickets==3
{'Beijing cuisine'}<=food_type
intercity_transport=={'train'}
{'natural scenery', 'Museum/Memorial Hall'}<=spot_type
{'Smart Room Control'}<=hotel_feature
hotel_price<=500
{'Beijing Quanjude (Qianmen Branch)'} <= restaurant_names
food_price<=100
transport_type<={'metro', 'taxi'}
{'The Palace Museum'}<=attraction_names
taxi_cars==1
answer:
[
"result=(day_count(plan)==2)",
"result=(people_count(plan)==3)",
"total_cost=0\nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=3000)",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=2: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro' and metro_tickets(activity_transports(activity))!=2: result=False",
"result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi' and taxi_cars(activity_transports(activity))!=1: result=False",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation' and accommodation_type(activity, target_city(plan))!='Smart Room Control': result=False\n  if activity_type(activity)=='accommodation' and activity_price(activity)>500: result=False",
"restaurant_type_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n    restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\nresult=({'Beijing cuisine'}<=restaurant_type_set)",
"attraction_type_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction':\n    attraction_type_set.add(attraction_type(activity, target_city(plan)))\nresult=({'natural scenery', 'Museum/Memorial Hall'}<=attraction_type_set)",
"intercity_transport_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['train', 'airplane']:\n    intercity_transport_set.add(activity_type(activity))\nresult=(intercity_transport_set=={'train'})",
"restaurant_names_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n    restaurant_names_set.add(activity_position(activity))\nresult=({'Beijing Quanjude (Qianmen Branch)'}<=restaurant_names_set)",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner'] and activity_price(activity)>100: result=False",
"inner_city_transportation_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='transportation': inner_city_transportation_set.add(activity_position(activity))\nresult=not({'walk'}&inner_city_transportation_set)",
"attraction_names_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction':\n    attraction_names_set.add(activity_position(activity))\nresult=({'The Palace Museum'}<=attraction_names_set)",
]
"""


def build_sl_trans_prompt(func_docs):
    return _SL_TRANS_HEAD + func_docs + _SL_TRANS_BODY


# ---------------------------------------------------------------------------
# Reflect prompt (mirrors nl2sl_hybrid_en.reflect_prompt)
# ---------------------------------------------------------------------------

_REFLECT_HEAD = """
下面提供了一组函数。请反思给定的 python 代码块，修复其中的错误，并按相同格式输出。
[
"python code block 1",
"python code block 2",
...
]
We offer functions below:"""

_REFLECT_BODY = """
请修复代码块中的错误，并以 json list 格式输出。
attractions_type、restaurants_type、accommodations_type 必须在上面给出的列表之内。若原类型不在列表中，必须把它改写为列表中!!!最相近的一个!!!。

activity_position(activity) 的返回值会被检查是否存在于数据库中。若某个名称被拒绝，请凭你自己的知识把它改成最相近的名称。同时 hotel_names 应当用 activity_position(activity) 检查，而不是 accommodation_type(activity, target_city(plan))；其他名称同理。
通常，对景点和餐厅，只要所需的那一项存在即满足要求；但住宿通常整个行程都住同一家酒店，所以需要检查规划中所有住宿活动。可以修改函数或取值使代码块正确。

修复规则：
- 执行器只暴露内建函数 `set`。len、bool、any、all、sum、str、int、float、map、sorted 未定义，使用会抛 NameError。非空判断改写为 result=(A&B)，判空/否定改写为 result=not(A&B)，子集判断用 A<=B，计数用循环内递增的计数变量。
- 每个代码块在全新的命名空间中独立运行：必须自包含并给 `result` 赋值；绝不引用其他块中定义的变量。"至少满足其一 / either"类分支必须合并成一条用 `or` 连接的代码块。
- 除非请求明确提到房间或床，绝不新增 room_count/room_type 或住宿约束。POI 名称保持与请求逐字一致。

你必须输出完整的代码块列表，包括那些本来就正确的约束。
The original code block is:
"""


def build_reflect_prompt(func_docs):
    return _REFLECT_HEAD + func_docs + _REFLECT_BODY


# ---------------------------------------------------------------------------
# Disjunction reflect turn (mirrors nl2sl_hybrid_en.disjunction_reflect_header)
# NOTE: formatted with .format(marker=..., k=...); literal braces are escaped.
# ---------------------------------------------------------------------------

disjunction_reflect_header = """
该请求包含一个"或"式需求（标记: "{marker}"，共 {k} 个编号分支）：规划只需满足这些编号分支中的至少一个，而不是全部。
规则：这类需求必须翻译成恰好一条自包含的 python 约束，在同一代码块内计算每个分支条件，并用 `or` 连接。
错误——折叠（禁止）：只把一个分支单独作为约束输出，会把"至少满足其一"变成无条件硬性要求。
错误——拆分（禁止）：把分支输出成多条独立约束意味着它们全部都必须满足（AND 语义）。
示例：
NL: '... must meet at least one of the following: 1. Do not wish to visit Xidan Commercial Street and Dingling Mausoleum; 2. Accommodation budget is 3300.0.'
正确的单条约束：
"attraction_name_set=set()\\nhotel_cost=0\\nfor activity in allactivities(plan):\\n  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\\n  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\\nbranch_1=not({{'Xidan Commercial Street', 'Dingling Mausoleum'}}&attraction_name_set)\\nbranch_2=(hotel_cost<=3300.0)\\nresult=(branch_1 or branch_2)"
执行器规则：只有内建函数 `set` 可用（没有 len/any/all/sum）；代码块必须完全自包含并给 `result` 赋值；POI 名称必须从请求中逐字复制。
可用函数如下："""

disjunction_reflect_tail = (
    "\n只重新输出修正后的析取约束本身：一个恰好包含一个字符串的 json list。"
    "\nThe request is:\n"
)
