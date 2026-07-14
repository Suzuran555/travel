"""Vendored zh-query NL->DSL path (port of nesy_agent/nl2sl_hybrid.py).

Carries the full rounds-1-4 hardening for CHINESE nature_language queries:
zh prompts/few-shots, shared language-agnostic normalizers + disjunction
verifier (zh reflect prompt text), hardened checker, coverage verifier.
Selected by the package's language router (lang_router.py) when the query
text is CJK-dominant; the English path stays nl2sl_hybrid_en.py.

Package-relative imports only; stock-checkout modules (ast_checker,
plan_for_check, concept_func, load_datasets, environment.language) are
byte-identical upstream and imported from the stock tree.
"""
import os
import sys
from json_repair import repair_json

project_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

import json
from copy import deepcopy
from chinatravel.symbol_verification.concept_func import func_dict
from .sv_compat import normalize_hard_logic_constraint
from .prompts_zh import NL2SL_INSTRUCTION
from chinatravel.agent.nesy_agent.ast_checker import HardLogicPyChecker
from chinatravel.data.load_datasets import save_json_file, load_json_file
from chinatravel.environment.language import city_names, normalize_lang


func_docs = """
(1) day_count(plan)
Docs: Get the number of days in the plan.
Return: int
(2) people_count(plan)
Docs: Get the number of people in the plan.
Return: int
(3) start_city(plan)
Docs: Get the start city of the plan.
Return: str
(4) target_city(plan)
Docs: Get the target city of the plan.
Return: str
(5) allactivities(plan)
Docs: Get all the activities in the plan.
Return: list of activities
(6) allactivities_count(plan)
Docs: Get the number of activities in the plan.
Return: int
(7) dayactivities(plan, day)
Docs: Get all the activities in the specific day [1, 2, 3, ...].
Return: list of activities
(8) activity_cost(activity)
Docs: Get the cost of specific activity without transport cost.
Return: float
(9) activity_position(activity)
Docs: Get the position name of specific activity.
Return: str
(10) activity_price(activity)
Docs: Get the price of specific activity. The price is price per person.
Return: float
(11) activity_type(activity)
Docs: Get the type of specific activity. ['breakfast', 'lunch', 'dinner', 'attraction', 'accommodation', 'train', 'airplane']
Return: str
(12) activity_tickets(activity)
Docs: Get the number of tickets needed for specific activity. ['attraction', 'train', 'airplane']
Return: int
(13) activity_transports(activity)
Docs: Get the transport information of specific activity.
Return: list of dict
(14) activity_start_time(activity)
Docs: Get the start time of specific activity.
Return: str
(15) activity_end_time(activity)
Docs: Get the end time of specific activity.
Return: str
(16) activity_time(activity)
Docs: Get the duration of specific activity.
Return: int (minutes)
(17) innercity_transport_cost(transports)
Docs: Get the total cost of innercity transport.
Return: float
(18) poi_recommend_time(city, poi):
Docs: Get the recommend time of specific poi in the city. Only support attractions now.
Return: int (minutes)
(19) poi_distance(city, poi1, poi2):
Docs: Get the distance between two pois in the city.
Return: float (km)
(20) innercity_transport_price(transports)
Docs: Get the price of innercity transport. The price is price per person.
Return: float
(21) innercity_transport_distance(transports)
Docs: Get the distance of innercity transport.
Return: float (km)
(22) metro_tickets(transports)
Docs: Get the number of metro tickets if the type of transport is metro.
Return: int
(23) taxi_cars(transports)
Docs: Get the number of taxi cars if the type of transport is taxi. The number of taxi cars is `(people_count(plan) + 3) // 4`.
Return: int
(24) room_count(activity)
Docs: Get the number of rooms of accommodation activity.
Return: int
(25) room_type(activity)
Docs: Get the type of room of accommodation activity.
1 for single room, 2 for double room. Must be 1 or 2. Never use "大床房" or "双床房" or other words but 1 or 2.
Return: int
(26) restaurant_type(activity, target_city)
Docs: Get the type of restaurant's cuisine in the target city. The return value must be in ['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜', '西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜', '湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理', '鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐'].
Return: str
(27) attraction_type(activity, target_city)
Docs: Get the type of attraction in the target city. The return value must be in ['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区'].
Return: str
(28) accommodation_type(activity, target_city)
Docs: Get the feature of accommodation in the target city to judge whether it's feature meets the user's requirement. The return value must be in ['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色 住宿', '茶室', '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房'].
Return: str
(29) innercity_transport_type(transports)
Docs: Get the type of innercity transport. The return value must be in ['metro', 'taxi', 'walk'].
Return: str
(30) innercity_transport_start_time(transports)
Docs: Get the start time of innercity transport.
Return: str
(31) innercity_transport_end_time(transports)
Docs: Get the end time of innercity transport.
Return: str
(32) intercity_transport_type(activity)
Docs: Get the type of intercity transport. The return value must be in ['train', 'airplane'].
Return: str
(33) innercity_transport_time(transports)
Docs: Get the duration of innercity transport.
Return: int (minutes)
(34) intercity_transport_origin(activity)
Docs: Get the origin city of intercity transport.
Return: str
(35) intercity_transport_destination(activity)
Docs: Get the destination city of intercity transport.
Return: str
"""

sl_trans_prompt = (
    """
下面提供了一组函数。请把自然语言约束翻译成 python 代码，并以 json list 格式输出。
（代码、函数名和 activity 类型等 DSL 取值保持英文；POI 名称、菜系、景点类型等数据取值保持与请求/上方列表一致的中文。）
variables:
(1) plan: a dict of the generated plan with information of the specific plan.

functions:"""
    + func_docs
    + """
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
2. 输出列表中的每个字符串都会在全新的命名空间中独立执行：它必须完全自包含并给 `result` 赋值。绝不能引用另一个字符串中定义的变量。"至少满足其一 / 任选其一"类需求必须写成一条约束：在同一代码块内计算所有子条件并用 `or` 连接。
3. 每个 POI 名称必须从请求中逐字复制为一个字符串：完整的原始子串，包括括号、'·'、分店后缀和空格。绝不在内部分隔符处拆分一个名称，绝不翻译、缩写或归一化。只有当明确的分隔符（顿号、逗号、"和"）分隔的是明显不同的场所时才拆分列表。
4. 永远输出基础约束，形式与示例完全一致：days、people、tickets 约束（attraction/airplane/train 票数和 metro 票数 == 人数）以及 taxi_cars 约束。除非请求明确提到房间、床、床型或酒店房型要求，绝不输出 room_count/room_type 或任何住宿房间约束：任何形式的 `room_count(activity)!=N` 或 `room_type(activity)!=N` 检查都不允许，无论是单独成条还是嵌在别的循环里。
5. 作答前自查：请求中的每个需求子句恰好映射为一条约束；除基础约束外，每条约束都能在请求中找到出处；没有出现被禁止的内建函数。

### 标准写法（严格照抄这些代码形状）
- 必须游览/就餐/入住 X：按正确的 activity 类型收集名称集合，然后 result=({'X'}<=name_set)
- "X、Y 里去一个 / 任选其一"：result=({'X','Y'}&name_set)（一条约束；用交集 &，不要用 <=）
- "不想去 / 避开 X"：result=not({'X'}&name_set)
- 市内交通方式偏好（不打车 / 不走路 / 只坐或尽量坐地铁 等）：一律把被禁止的方式写成如下黑名单形式（"只坐地铁"禁止 walk 和 taxi）：
"inner_city_transportation_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='transportation': inner_city_transportation_set.add(activity_position(activity))\nresult=not({'walk', 'taxi'}&inner_city_transportation_set)"
绝不要用 innercity_transport_type 或 activity_transports 表达方式偏好，也绝不要把黑名单改写成白名单。
- "在 A 到 B 之间游览 X"：该活动必须覆盖整个时间窗，即开始不晚于 A 且结束不早于 B。只按 activity_position 匹配：
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

### 析取（"至少满足以下条件之一"、"满足任意一条"、"任选其一"）
这类请求列出若干编号分支，但规划只需满足其中一条。必须翻译成恰好一条自包含约束：在同一代码块内计算每个分支条件，并用 `or` 连接。
NL: '……必须满足以下要求中的至少一个：1. 不想去西单商业街和定陵；2. 住宿预算为3300.0。'
正确（一条约束）：
"attraction_name_set=set()\nhotel_cost=0\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\n  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\nbranch_1=not({'西单商业街', '定陵'}&attraction_name_set)\nbranch_2=(hotel_cost<=3300.0)\nresult=(branch_1 or branch_2)"
错误——折叠（禁止）：只输出一个分支，例如只写 "result=(hotel_cost<=3300.0)"，会把"至少满足其一"变成无条件硬性要求。
错误——拆分（禁止）：把 branch_1 和 branch_2 输出为两条列表项意味着两者都必须满足（AND 语义）。

### 注意!!!
如果自然语言约束中出现上面函数未定义的伪代码，必须用上面提供的函数把它改写成 python 代码块。通常，对景点和餐厅，只要所需的那一项出现即满足要求；但住宿通常整个行程都住同一家酒店，所以需要检查规划中所有住宿活动。
###

如果发现自然语言约束本身有错误，需要在代码块中修正。例如出现 {'自然景观'} <= spot_type 时，应改用上面提供的 '自然风光'；同理 {'大学'} <= spot_type 应改为 '大学校园'，'繁华的商业街' 应改为 '商业街区'。restaurant_type 和 accommodation_type 的取值同样如此处理。

Example:
nature_language:
days==2
people_number==3
cost<=3000
tickets==3
{'北京菜'}<=food_type
intercity_transport=={'train'}
{'自然风光', '博物馆/纪念馆'}<=spot_type
{'智能客控'}<=hotel_feature
hotel_price<=500
{'北京全聚德(前门店)'} <= restaurant_names
food_price<=100
transport_type<={'metro', 'taxi'}
{'故宫博物院'}<=attraction_names
taxi_cars==1
answer:
[
"result=(day_count(plan)==2)",
"result=(people_count(plan)==3)",
"total_cost=0\nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=3000)",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=2: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro' and metro_tickets(activity_transports(activity))!=2: result=False",
"result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi' and taxi_cars(activity_transports(activity))!=1: result=False",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation' and accommodation_type(activity, target_city(plan))!='智能客控': result=False\n  if activity_type(activity)=='accommodation' and activity_price(activity)>500: result=False",
"restaurant_type_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n    restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\nresult=({'北京菜'}<=restaurant_type_set)",
"attraction_type_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction':\n    attraction_type_set.add(attraction_type(activity, target_city(plan)))\nresult=({'自然风光', '博物馆/纪念馆'}<=attraction_type_set)",
"intercity_transport_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['train', 'airplane']:\n    intercity_transport_set.add(activity_type(activity))\nresult=(intercity_transport_set=={'train'})",
"restaurant_names_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n    restaurant_names_set.add(activity_position(activity))\nresult=({'北京全聚德(前门店)'}<=restaurant_names_set)",
"result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['breakfast', 'lunch', 'dinner'] and activity_price(activity)>100: result=False",
"inner_city_transportation_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='transportation': inner_city_transportation_set.add(activity_position(activity))\nresult=not({'walk'}&inner_city_transportation_set)",
"attraction_names_set = set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='attraction':\n    attraction_names_set.add(activity_position(activity))\nresult=({'故宫博物院'}<=attraction_names_set)",
]
"""
)

reflect_prompt = (
    """
下面提供了一组函数。请反思给定的 python 代码块，修复其中的错误，并按相同格式输出。
[
"python code block 1",
"python code block 2",
...
]
We offer functions below:"""
    + func_docs
    + """
请修复代码块中的错误，并以 json list 格式输出。
attractions_type、restaurants_type、accommodations_type 必须在上面给出的列表之内。若原类型不在列表中，必须把它改写为列表中!!!最相近的一个!!!。例如 '购物街' 改为 '商业街区'；'本地特色菜' 通常指该城市的本地菜系。
activity_position(activity) 的返回值会被检查是否存在于数据库中。若某个名称被拒绝，请凭你自己的知识把它改成最相近的名称。例如 '故宫' 改为 '故宫博物院'；'A附近的B' 可能是 'A(B店)' 或 'A（B店）' 等类似形式。
同时 hotel_names 应当用 activity_position(activity) 检查，而不是 accommodation_type(activity, target_city(plan))；其他名称同理。
通常，对景点和餐厅，只要所需的那一项存在即满足要求；但住宿通常整个行程都住同一家酒店，所以需要检查规划中所有住宿活动。可以修改函数或取值使代码块正确。

修复规则：
- 执行器只暴露内建函数 `set`。len、bool、any、all、sum、str、int、float、map、sorted 未定义，使用会抛 NameError。非空判断改写为 result=(A&B)，判空/否定改写为 result=not(A&B)，子集判断用 A<=B，计数用循环内递增的计数变量。
- 每个代码块在全新的命名空间中独立运行：必须自包含并给 `result` 赋值；绝不引用其他块中定义的变量。"至少满足其一 / 任选其一"类分支必须合并成一条用 `or` 连接的代码块。
- 除非请求明确提到房间或床，绝不新增 room_count/room_type 或住宿约束。POI 名称保持与请求逐字一致。

你必须输出完整的代码块列表，包括那些本来就正确的约束。
The original code block is:
"""
)


# ---------------------------------------------------------------------------
# Hardening ported from the (rounds 1-3 validated) English module
# nl2sl_hybrid_en.py. The normalizers and the disjunction verifier operate on
# the emitted DSL, which is shared between the two paths (python code, English
# function names); their NL-facing guards (room-strip regex, disjunction
# markers) already carry Chinese terms. Only the reflect-turn prompt text is
# re-rendered in Chinese here.
# ---------------------------------------------------------------------------
from .nl2sl_hybrid_en import (
    normalize_generated_constraints,
    enforce_disjunction,
)
from .constraint_coverage import enforce_coverage

disjunction_reflect_header_zh = """
该请求包含一个"或"式需求（标记: "{marker}"，共 {k} 个编号分支）：规划只需满足这些编号分支中的至少一个，而不是全部。
规则：这类需求必须翻译成恰好一条自包含的 python 约束，在同一代码块内计算每个分支条件，并用 `or` 连接。
错误——折叠（禁止）：只把一个分支单独作为约束输出，会把"至少满足其一"变成无条件硬性要求。
错误——拆分（禁止）：把分支输出成多条独立约束意味着它们全部都必须满足（AND 语义）。
示例：
NL: '……必须满足以下要求中的至少一个：1. 不想去西单商业街和定陵；2. 住宿预算为3300.0。'
正确的单条约束：
"attraction_name_set=set()\\nhotel_cost=0\\nfor activity in allactivities(plan):\\n  if activity_type(activity)=='attraction': attraction_name_set.add(activity_position(activity))\\n  if activity_type(activity)=='accommodation': hotel_cost+=activity_cost(activity)\\nbranch_1=not({{'西单商业街', '定陵'}}&attraction_name_set)\\nbranch_2=(hotel_cost<=3300.0)\\nresult=(branch_1 or branch_2)"
执行器规则：只有内建函数 `set` 可用（没有 len/any/all/sum）；代码块必须完全自包含并给 `result` 赋值；POI 名称必须从请求中逐字复制。
可用函数如下："""

disjunction_reflect_tail_zh = (
    "\n只重新输出修正后的析取约束本身：一个恰好包含一个字符串的 json list。"
    "\nThe request is:\n"
)


def make_checker(target_city):
    """HardLogicPyChecker with the canonical intra-city-mode idiom whitelisted.

    The oracle idiom for transport-mode preferences filters on
    activity_type(activity)=='transportation' (vacuously true on real plans).
    Without this whitelist the AST value-checker flags 'transportation' as
    invalid and the reflect loop burns retries mangling the canonical pattern.
    """
    checker = HardLogicPyChecker(target_city)
    tracker = checker.trackers.get("activity_type")
    if tracker is not None:
        tracker.valid_values.add("transportation")
    return checker


def load_example_plans(example_plans_dir=os.path.join(
        project_path, "chinatravel", "agent", "nesy_agent", "plan_for_check")):
    plan_for_test = {}
    plan_files = os.listdir(example_plans_dir)
    plan_files = [plan_file for plan_file in plan_files if plan_file.endswith(".json")]
    for file in plan_files:
        with open(os.path.join(example_plans_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            plan_for_test[int(file.split("day")[1].split(".")[0])] = data
    return plan_for_test


EXAMPLE_PLANS = load_example_plans()


def get_first_list_in_str(json_str):
    # 使用栈得到第一个合法的list
    json_str = repair_json(json_str, ensure_ascii=False)
    st = 0
    # print(json_str)
    while st < len(json_str) and json_str[st] != "[":
        st += 1
    json_str = json_str[st:]
    stack = []
    for i, c in enumerate(json_str):
        if c == "[":
            stack.append(i)
        elif c == "]":
            stack.pop()
            if not stack:
                res = json_str[: i + 1]
                return res
    return "[]"


def _nl2sl_instruction(lang):
    if normalize_lang(lang) == "en":
        from .prompts_en import NL2SL_INSTRUCTION as EN_NL2SL_INSTRUCTION

        return EN_NL2SL_INSTRUCTION
    return NL2SL_INSTRUCTION


def nl2sl_step1(query, backbone_llm, lang="zh"):

    nature_language = query["nature_language"]
    messages = [{"role": "user", "content": _nl2sl_instruction(lang).format(nature_language)}]
    # print(messages[0]["content"])
    query_ = backbone_llm(messages, one_line=False, json_mode=True)

    try:
        l_ptr = query_.find("{")
        r_ptr = query_.rfind("}")
        if l_ptr != -1 and r_ptr != -1:
            query_ = query_[l_ptr : r_ptr + 1]

        query_ = json.loads(query_)
        for key in query_:
            query[key] = query_[key]
    except Exception as e:
        query["hard_logic"] = []
        return query


    return query


def nl2sl_step2(query, backbone_llm):
    try:
        query["hard_logic"] = [str(hl) for hl in query["hard_logic"]]
        hard_logic = "\n".join(query["hard_logic"])
    except Exception as e:
        query["hard_logic"] = []
        query["hard_logic_py"] = []
        return query
    messages = [
        {
            "role": "user",
            "content": sl_trans_prompt
            + hard_logic
            + "\n The query is: \n"
            + query["nature_language"]
            + "\nanswer:\n",
        }
    ]
    # print(messages[0]["content"])
    hard_logic_py = backbone_llm(messages, one_line=False, json_mode=True)
    # l_ptr = hard_logic_py.find("[")
    # r_ptr = hard_logic_py.rfind("]")
    # if l_ptr != -1 and r_ptr != -1:
    #     hard_logic_py = hard_logic_py[l_ptr : r_ptr + 1]
    hard_logic_py = get_first_list_in_str(hard_logic_py)
    # print(hard_logic_py)
    try:
        query["hard_logic_py"] = json.loads(hard_logic_py)
    except Exception as e:
        query["error_hard_logic_py"] = hard_logic_py
        query["hard_logic_py"] = []
    query["hard_logic_py"] = [str(item) for item in query["hard_logic_py"]]
    query["hard_logic_py"] = list(set(query["hard_logic_py"]))
    query["hard_logic_py"] = normalize_generated_constraints(
        query["hard_logic_py"],
        query.get("people_number"),
        query.get("nature_language"),
    )
    return query


def check(query):
    run_error_list = []
    run_error_idx = []
    hard_logic_py = query["hard_logic_py"]

    if query["days"] not in EXAMPLE_PLANS:
        print("Error: days should be in [1, 2, 3, 4, 5, 6]")
        return [], []

    example_plan = EXAMPLE_PLANS[query["days"]]
    for idx, constraint in enumerate(hard_logic_py):
        vars_dict = deepcopy(func_dict)
        vars_dict["plan"] = example_plan
        try:
            # Evaluate the constraint in a safe manner
            exec(
                normalize_hard_logic_constraint(constraint),
                {
                    "__builtins__": {
                        "set": set,
                    }
                },
                vars_dict,
            )
        except Exception as e:
            if str(e) not in [
                "Failed to create Point instance from string: unknown format.",
            ]:
                run_error_list.append(str(e))
                run_error_idx.append(idx)
    return run_error_list, run_error_idx


def reflect_info(query, checker: HardLogicPyChecker):
    hard_logic_py = query["hard_logic_py"]

    run_error_list, run_error_idx = check(query)
    if len(run_error_list):
        return run_error_list, run_error_idx, [], []
    value_error_list = [checker.check(constraint)[0] for constraint in hard_logic_py]
    value_error_idx = [idx for idx, item in enumerate(value_error_list) if len(item)]
    value_error_list = [item for sublist in value_error_list for item in sublist]
    return run_error_list, run_error_idx, value_error_list, value_error_idx


def reflect(query, backbone_llm, run_error_list, value_error_list):

    content = (
        reflect_prompt
        + str(query["hard_logic_py"])
        + "The error is: "
        + "\n".join(run_error_list)
        + "\n".join(value_error_list)
        + "\nThe query is: \n"
        + query["nature_language"]
        + "\nanswer:\n"
    )
    # print(content)
    messages = [{"role": "user", "content": content}]
    res = backbone_llm(messages, one_line=False, json_mode=True)
    # l_ptr = res.find("[")
    # r_ptr = res.rfind("]")
    # if l_ptr != -1 and r_ptr != -1:
    #     res = res[l_ptr : r_ptr + 1]
    res = get_first_list_in_str(res)
    # print(res)
    try:
        query["hard_logic_py"] = json.loads(res)
    except Exception as e:
        query["error_hard_logic_py"] = res
        query["hard_logic_py"] = []
    query["hard_logic_py"] = [str(item) for item in query["hard_logic_py"]]
    query["hard_logic_py"] = normalize_generated_constraints(
        query["hard_logic_py"],
        query.get("people_number"),
        query.get("nature_language"),
    )
    # print(query["hard_logic_py"])
    return query, len(run_error_list + value_error_list) == 0


def nl2sl_step3(query, backbone_llm, checker, max_trails=5):

    cnt = 0
    query["reflect_info"] = []
    query["hard_logic_py_ood"] = []
    value_error_idx = []
    run_error_idx = []
    while cnt < max_trails:
        run_error_list, run_error_idx, value_error_list, value_error_idx = reflect_info(
            query, checker
        )
        query["reflect_info"].append(
            {
                "cnt": cnt,
                "run_error_list": run_error_list,
                "value_error_list": value_error_list,
                "hard_logic_py": query["hard_logic_py"],
            }
        )
        flag = len(run_error_list + value_error_list) == 0
        if flag:
            break
        query, _ = reflect(query, backbone_llm, run_error_list, value_error_list)
        query["hard_logic_py"] = list(set(query["hard_logic_py"]))

        # if "OOD!!!" in query["hard_logic_py"]:
        #     run_error_list, run_error_idx, value_error_list, value_error_idx = (
        #         reflect_info(query, checker)
        #     )
        #     query["ood"] = True
        #     ood_idx = list(set(run_error_idx + value_error_idx))
        #     for idx in ood_idx:
        #         query["hard_logic_py_ood"].append(query["hard_logic_py"][idx])
        #     for ood_logic in query["hard_logic_py_ood"]:
        #         query["hard_logic_py"].remove(ood_logic)
        #     return query

        cnt += 1
    query["reflect_cnt"] = cnt
    run_error_list, run_error_idx, value_error_list, value_error_idx = reflect_info(
        query, checker
    )
    query["reflect_info"].append(
        {
            "cnt": cnt,
            "run_error_list": run_error_list,
            "value_error_list": value_error_list,
            "hard_logic_py": query["hard_logic_py"],
        }
    )
    error_indices = set(run_error_idx + value_error_idx)
    query["hard_logic_py"] = [
        val
        for idx, val in enumerate(query["hard_logic_py"])
        if idx not in error_indices
    ]
    # mechanical disjunction verifier (ported from the en module): runs on the
    # final surviving list so a missing OR-constraint is re-requested even when
    # the reflect loop above converged without errors; zh reflect prompt text
    query = enforce_disjunction(
        query,
        backbone_llm,
        header=disjunction_reflect_header_zh,
        tail=disjunction_reflect_tail_zh,
    )
    # round-4 coverage / span-grounding verifier (zh DSL literal tables):
    # deterministic NL-triggered fixes for dropped (local-cuisine,
    # 机票->airplane, budget), invented (taxi-car scaling, ungrounded cost
    # caps) and wrong (explicit room count, 或者-category expansion)
    # constraints
    query = enforce_coverage(query, lang="zh")
    # ood_idx = list(set(run_error_idx + value_error_idx))
    # if len(ood_idx):
    #     query["ood"] = True
    #     for idx in ood_idx:
    #         query["hard_logic_py_ood"].append(query["hard_logic_py"][idx])
    #     for ood_logic in query["hard_logic_py_ood"]:
    #         query["hard_logic_py"].remove(ood_logic)
    return query


def nl2sl(query, backbone_llm, checker, cache_dir="cache_hybrid", lang="zh"):
    file_path = os.path.join(
        project_path,
        cache_dir,
        "translation_{}_reflect".format(backbone_llm.name),
        "{}.json".format(query["uid"]),
    )
    if os.path.exists(file_path):
        query = load_json_file(file_path)
        return query
    city_list = city_names(lang)
    if query["target_city"] not in city_list or query["start_city"] not in city_list:
        query["hard_logic"] = []
        query["hard_logic_py"] = []
        query["ood"] = True
        return query
    query = nl2sl_step1(query, backbone_llm, lang=lang)
    query = nl2sl_step2(query, backbone_llm)
    query = nl2sl_step3(query, backbone_llm, checker)

    save_json_file(query, file_path)
    return query


def nl2sl_reflect(query, backbone_llm, lang="zh"):
    lang = normalize_lang(lang)
    # package tweak: `lang` here is the QUERY language chosen by the router,
    # while the city fields carry the ENVIRONMENT language's spelling -- a
    # zh-text query in an en environment must not be falsely marked OOD, so
    # the gate accepts either language's city list
    city_list = set(city_names("zh")) | set(city_names("en"))
    if "target_city" in query and "start_city" in query:
        if query["target_city"] not in city_list or query["start_city"] not in city_list:
            query["hard_logic"] = []
            query["hard_logic_py"] = []
            query["ood"] = True
            return query
    query = nl2sl_step1(query, backbone_llm, lang=lang)
    query = nl2sl_step2(query, backbone_llm)
    try:
        checker = make_checker(query["target_city"])
        query = nl2sl_step3(query, backbone_llm, checker)
        query["hard_logic_py_iter_3"] = query["hard_logic_py"]
    except Exception:
        import traceback

        query["reflect_error"] = traceback.format_exc()
    return query
