r"""매핑 평가셋 v1 생성 (결정적, 난수 없음).

실행: decision-supporter 의 가상환경으로 실행한다 (현재 백엔드 별칭 규칙을 그대로 import 하기 위함).
    cd decision-supporter/backend
    ..\.venv\Scripts\python.exe ..\..\retriever-eval\scripts\build_dataset.py

산출 (retriever-eval/data):
    fields.json          entity 별 필드, 설명, 별칭(백엔드 mapping.py 에서 그대로 추출)
    header_queries.csv   과제 A: 열 이름 -> 필드   (정답 필드 또는 NONE)
    master_items.csv     과제 B: 합성 품목 마스터 (제품·자재)
    value_queries.csv    과제 B: 품목 표기 -> 마스터 (정답 코드 / NONE / AMBIG)

모든 값은 합성이다. 실제 기업 데이터가 아니다.
정답은 사람이(이 스크립트 작성 시) 정했고, 어떤 검색 방식의 출력도 보지 않고 정했다.
한계: 평가셋 작성자가 백엔드 별칭 목록을 본 상태에서 만들었다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys

BACKEND = pathlib.Path(__file__).resolve().parents[2] / "decision-supporter" / "backend"
sys.path.insert(0, str(BACKEND))
from app.ingest.mapping import ENTITY_FIELDS, field_for_header, norm_header  # noqa: E402
from app.ingest.validation import _variant_key, norm_name  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

# ───────────────────────── 필드 설명 (검색 대상 문서) ─────────────────────────
# 데이터 계약 v1 문서의 의미를 한 줄로 옮긴 것. 별칭은 mapping.py 에서 그대로 가져온다.
FIELD_DESC = {
    "monthly_demand": {
        "product_id": "제품 코드. 제품을 식별하는 번호",
        "product_name": "제품 이름",
        "period": "수요가 발생한 월. 연월 YYYY-MM",
        "demand_qty": "그 달의 수요 수량. 판매·주문·출고된 양",
        "unit": "수량의 단위",
    },
    "product": {
        "product_id": "제품 코드. 제품을 식별하는 번호",
        "product_name": "제품 이름",
        "base_unit": "제품의 기준 단위. 재고·생산을 세는 단위",
        "product_type": "제품 유형. 완제품인지 반제품인지",
        "safety_stock_qty": "안전재고 수량. 최소로 유지할 재고",
    },
    "material": {
        "material_id": "자재 코드. 원자재·부품을 식별하는 번호",
        "material_name": "자재 이름",
        "base_unit": "자재의 기준 단위",
        "safety_stock_qty": "안전재고 수량. 최소로 유지할 재고",
    },
    "bom": {
        "bom_id": "BOM 코드. 자재명세서를 식별하는 번호",
        "product_id": "만드는 제품의 코드. 상위 품목, 모품목",
        "bom_version": "BOM 버전. 개정 차수",
        "status": "BOM 승인 상태. 승인 또는 초안",
        "is_active": "활성 여부. 현재 사용하는 BOM 인지",
        "valid_from": "유효 시작일. 적용을 시작하는 날짜",
        "valid_to": "유효 종료일. 적용이 끝나는 날짜",
        "base_qty": "기준 수량. 소요량을 정의한 기준 생산 수량",
        "line_no": "구성품 줄 번호. 순번",
        "component_type": "구성품 유형. 자재인지 반제품인지",
        "component_id": "구성품 코드. 투입되는 자재나 반제품, 하위 품목, 자품목",
        "qty_per": "소요량. 기준 수량당 들어가는 구성품의 양",
        "unit": "소요량의 단위",
        "scrap_rate": "스크랩 비율. 손실률",
    },
    "inventory": {
        "item_type": "품목 구분. 제품인지 자재인지",
        "item_id": "품목 코드. 재고를 가진 제품 또는 자재의 번호",
        "as_of_date": "재고 기준일. 재고를 센 날짜",
        "on_hand_qty": "현재고. 지금 보유한 수량",
        "allocated_qty": "할당 재고. 이미 예약된 수량",
        "available_qty": "가용 재고. 현재고에서 할당을 뺀 수량",
        "unit": "수량의 단위",
    },
}

# ───────────────────────── 과제 A: 손으로 쓴 헤더 ─────────────────────────
# 형식: "헤더=>정답필드" (정답이 없으면 NONE). 범주:
#   synonym  별칭 목록에 없는 동의어·영문 표기
#   typo     약어·오타·ERP 식 축약
#   compound 수식어가 붙은 복합 표기
#   none     그 데이터 종류의 어떤 필드도 아닌 열 (매핑하면 안 됨)
#   trap     그럴듯하지만 매핑하면 안 되는 열 (다른 의미의 코드·수량·날짜)
HAND = {
    "monthly_demand": {
        "synonym": """품목번호=>product_id|상품코드=>product_id|Item No.=>product_id|Part No=>product_id|품목명칭=>product_name
            |상품명=>product_name|제품 이름=>product_name|Description=>product_name|납기년월=>period|판매월=>period|대상월=>period
            |연월=>period|출고월=>period|판매수량=>demand_qty|출고수량=>demand_qty|오더수량=>demand_qty|수주량=>demand_qty
            |수주수량=>demand_qty|Sales Qty=>demand_qty|월 판매 실적=>demand_qty|계량단위=>unit|Unit of Measure=>unit""",
        "typo": """PROD_CD=>product_id|품목 CD=>product_id|제폼코드=>product_id|PRD_NM=>product_name|제품몀=>product_name
            |기준년월=>period|YM=>period|수요랑=>demand_qty|DMD_QTY=>demand_qty|판매 수랑=>demand_qty|단 위=>unit""",
        "compound": """2026년 월별 수요량=>demand_qty|당월 수요=>demand_qty|월별 판매수량 합계=>demand_qty|고객 주문수량=>demand_qty
            |수요 예측 대상 월=>period|판매 기준월=>period|ERP 품번=>product_id|사내 제품코드=>product_id|완제품 코드=>product_id
            |제품 한글명=>product_name|수량 단위 표기=>unit""",
        "none": "비고|담당자|거래처명|단가|금액|영업소|작성일|순번|납품처|통화",
        "trap": "생산량|재고량|자재코드|거래처코드|고객코드|반품수량|예측수량|전년동월 수요|주문번호|창고코드|제품군|규격|목표수량",
    },
    "product": {
        "synonym": """품목번호=>product_id|상품코드=>product_id|모델코드=>product_id|제품 명칭=>product_name|모델명=>product_name
            |상품명=>product_name|재고단위=>base_unit|관리단위=>base_unit|품목계정=>product_type|완/반제품 구분=>product_type
            |최소재고=>safety_stock_qty|Min Stock=>safety_stock_qty|최소 유지 재고=>safety_stock_qty""",
        "typo": """PRD_CD=>product_id|ITEM_NO=>product_id|제품몀=>product_name|BASE UOM=>base_unit|SS_QTY=>safety_stock_qty
            |안전제고=>safety_stock_qty|PROD_TYPE=>product_type""",
        "compound": """완제품/반제품 여부=>product_type|사내 품목코드=>product_id|안전재고 기준량=>safety_stock_qty|대표 품명=>product_name
            |재고 관리 기준단위=>base_unit|제품 유형 구분값=>product_type""",
        "none": "비고|단가|표준원가|등록일|사용여부|공급사|규격|중량|바코드|담당자",
        "trap": "자재코드|현재고|구매단위|판매단가|제품군코드|리드타임|최대재고|포장단위|거래처 품번|대체품 코드|발주점",
    },
    "material": {
        "synonym": """원재료코드=>material_id|부품코드=>material_id|자재번호=>material_id|Material No.=>material_id
            |원부자재명=>material_name|자재 명칭=>material_name|부품명=>material_name|재료명=>material_name|재고단위=>base_unit
            |관리단위=>base_unit|최소재고=>safety_stock_qty|Min Stock=>safety_stock_qty""",
        "typo": """MAT_CD=>material_id|자제코드=>material_id|MTRL_NM=>material_name|자재몀=>material_name|BASE UOM=>base_unit
            |안전제고=>safety_stock_qty|SS_QTY=>safety_stock_qty""",
        "compound": """원자재 품목코드=>material_id|사내 자재코드=>material_id|구매 자재명=>material_name|자재 기준단위=>base_unit
            |자재 안전재고 수량=>safety_stock_qty|원재료 한글명=>material_name""",
        "none": "비고|단가|공급사|리드타임|규격|등록일|보관위치|최소발주량|통화|담당자",
        "trap": "제품코드|공급사 코드|구매단위|발주단위|현재고|공급사 자재코드|대체자재코드|자재그룹|최대재고|발주점|입고예정일",
    },
    "bom": {
        "synonym": """BOM No.=>bom_id|배합표 번호=>bom_id|모품번=>product_id|완제품코드=>product_id|생산품목코드=>product_id
            |개정번호=>bom_version|차수=>bom_version|결재상태=>status|승인여부=>status|사용 유무=>is_active|적용여부=>is_active
            |시작일자=>valid_from|적용일=>valid_from|효력발생일=>valid_from|만료일=>valid_to|종료일자=>valid_to|적용종료=>valid_to
            |기준생산수량=>base_qty|배치수량=>base_qty|항목번호=>line_no|Seq=>line_no|자재/반제품 구분=>component_type
            |투입품 유형=>component_type|자품번=>component_id|자품목코드=>component_id|투입자재코드=>component_id
            |원재료코드=>component_id|투입량=>qty_per|원단위=>qty_per|필요수량=>qty_per|사용량=>qty_per|소요단위=>unit
            |로스율=>scrap_rate|감모율=>scrap_rate|Loss Rate=>scrap_rate""",
        "typo": """BOM_VER=>bom_version|소요랑=>qty_per|COMP_CD=>component_id|스크렙률=>scrap_rate|VALID_FR=>valid_from
            |VALID_TO_DT=>valid_to|ACTIVE_YN=>is_active|PARENT_ITEM_CD=>product_id""",
        "compound": """구성품 소요량 합계=>qty_per|상위 제품코드=>product_id|하위 자재코드=>component_id|BOM 적용 시작일=>valid_from
            |BOM 유효 종료일=>valid_to|BOM 개정 버전=>bom_version|구성품 투입 단위=>unit|공정 손실률=>scrap_rate""",
        "none": "비고|작성자|승인자|공정코드|작업장|단가|등록일시|도면번호|공급사",
        "trap": "수율|구성품명|상위품목명|대체자재|공정순서|리드타임|승인일|개정일자|자재 단가|현재고|작업시간",
    },
    "inventory": {
        "synonym": """제품/자재 구분=>item_type|품목 종류=>item_type|품목번호=>item_id|자재번호=>item_id|SKU=>item_id
            |재고일자=>as_of_date|조사일=>as_of_date|실사일자=>as_of_date|스냅샷 일자=>as_of_date|장부재고=>on_hand_qty
            |실재고=>on_hand_qty|보유량=>on_hand_qty|Stock Qty=>on_hand_qty|재고수량=>on_hand_qty|예약수량=>allocated_qty
            |기할당수량=>allocated_qty|Reserved Qty=>allocated_qty|사용가능재고=>available_qty|가용량=>available_qty
            |Free Stock=>available_qty|계량단위=>unit""",
        "typo": """현제고=>on_hand_qty|ALLOC_QTY=>allocated_qty|AVAIL_QTY=>available_qty|ITEM_CD=>item_id|ONHAND_QTY=>on_hand_qty
            |STK_DT=>as_of_date|ITEM_TP=>item_type|할당제고=>allocated_qty""",
        "compound": """당일 현재고=>on_hand_qty|할당된 재고 수량=>allocated_qty|창고 보유 재고 수량=>on_hand_qty|출고 가능한 가용 수량=>available_qty
            |재고 조사 기준일자=>as_of_date|재고 품목 구분값=>item_type|재고 품목코드=>item_id""",
        "none": "비고|창고|로케이션|담당자|단가|재고금액|LOT번호|유통기한|등록일시",
        "trap": "안전재고|입고예정수량|발주잔량|불량재고|입고일|출고수량|품목명|창고코드|전월재고|적정재고|검사대기 수량",
    },
}


def split_of(key: str) -> str:
    """안정 해시로 dev(40%) / test(60%) 분할. 임계값은 dev 에서만 정한다."""
    return "dev" if int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10 < 4 else "test"


def build_fields() -> dict:
    out = {}
    for entity, fields in ENTITY_FIELDS.items():
        out[entity] = {}
        for fname, spec in fields.items():
            out[entity][fname] = {"desc": FIELD_DESC[entity][fname], "required": spec["required"],
                                  "strong": list(spec["strong"]), "weak": list(spec["weak"])}
    return out


def build_header_queries() -> list[dict]:
    rows, seen = [], set()

    def add(entity, header, gold, category):
        key = (entity, header)
        if key in seen:
            return
        seen.add(key)
        rule = field_for_header(header, entity)
        # 범주는 '실제로 현재 규칙이 아는 표기인가'로 다시 판정한다 (작성자의 분류 실수를 막기 위함)
        if category in ("synonym", "typo", "compound") and rule == gold:
            category = "alias_format"
        rows.append({"qid": f"A{len(rows) + 1:04d}", "entity": entity, "header": header, "gold": gold, "category": category,
                     "rule_pred": rule or "", "split": split_of(f"A|{entity}|{header}")})

    for entity, fields in ENTITY_FIELDS.items():
        # 1) 별칭 그대로 (기준선이 반드시 맞혀야 하는 확인용)
        for fname, spec in fields.items():
            for alias in spec["strong"] + spec["weak"]:
                add(entity, alias, fname, "alias_exact")
        # 2) 별칭의 표기 변형 (괄호 단위, 띄어쓰기, 대문자): 정규화로 풀려야 함
        for fname, spec in fields.items():
            for alias in spec["strong"][:2]:
                if alias.isascii():
                    add(entity, alias.upper().replace("_", " "), fname, "alias_format")
                else:
                    add(entity, " ".join(alias) if len(alias) <= 3 else alias[:2] + " " + alias[2:], fname, "alias_format")
                    add(entity, f"{alias}(기준)", fname, "alias_format")
        # 3) 손으로 쓴 범주
        for category, text in HAND[entity].items():
            for item in text.replace("\n", "").split("|"):
                item = item.strip()
                if not item:
                    continue
                header, _, gold = item.partition("=>")
                add(entity, header.strip(), (gold.strip() or "NONE"), category)
    return rows


# ───────────────────────── 과제 B: 합성 품목 마스터 ─────────────────────────
MASTER = """
product|P-1001|백상지 80g A4 500매
product|P-1002|백상지 100g A4 500매
product|P-1003|백상지 80g A3 500매
product|P-1004|복사용지 75g A4 2500매 박스
product|P-1005|아트지 150g 국전
product|P-1006|아트지 200g 국전
product|P-1007|스노우지 150g 국전
product|P-1008|크라프트지 120g 롤 1000mm
product|P-1009|크라프트지 80g 롤 1000mm
product|P-1010|화장지 원단 2겹 30m
product|P-1011|화장지 원단 3겹 30m
product|P-1012|골판지 원단 SW A골
product|P-1013|골판지 원단 DW AB골
product|MP-2001|샌드위치 패널 EPS 50T 벽체용
product|MP-2002|샌드위치 패널 EPS 75T 벽체용
product|MP-2003|샌드위치 패널 EPS 100T 지붕용
product|MP-2004|샌드위치 패널 우레탄 50T 벽체용
product|MP-2005|샌드위치 패널 글라스울 75T 벽체용
product|MP-2006|메탈 패널 알루미늄 3T 불소도장
product|MP-2007|메탈 패널 아연도 1.2T
product|MP-2008|스테인리스 패널 SUS304 1.2T 헤어라인
product|MP-2009|스테인리스 패널 SUS304 1.5T 헤어라인
product|MP-2010|스테인리스 패널 SUS316 1.2T 미러
product|MP-2011|징크 패널 0.7T 돌출이음
product|MP-2012|타공 패널 알루미늄 2T
product|PK-3001|골판지 상자 A형 300x200x150
product|PK-3002|골판지 상자 A형 400x300x200
product|PK-3003|골판지 상자 B형 300x200x150
product|PK-3004|컬러박스 화장품용 소
product|PK-3005|컬러박스 화장품용 대
product|PK-3006|쇼핑백 크라프트 중
product|PK-3007|쇼핑백 크라프트 대
product|PK-3008|라벨 스티커 원형 50mm 유광
product|PK-3009|라벨 스티커 원형 50mm 무광
product|PK-3010|리플렛 A4 3단 접지
product|PK-3011|카탈로그 A4 16p 중철
product|PK-3012|포장 완충재 펄프몰드 6구
material|M-0101|표백 화학펄프 활엽수 BHKP
material|M-0102|표백 화학펄프 침엽수 BSKP
material|M-0103|폐지 OCC 국내산
material|M-0104|폐지 ONP 신문고지
material|M-0105|탄산칼슘 충전제
material|M-0106|전분 사이징제
material|M-0201|컬러강판 0.5T 백색
material|M-0202|컬러강판 0.5T 은회색
material|M-0203|컬러강판 0.6T 백색
material|M-0204|EPS 단열재 50T 비드법 2종
material|M-0205|EPS 단열재 75T 비드법 2종
material|M-0206|우레탄 원액 폴리올
material|M-0207|우레탄 원액 MDI
material|M-0208|글라스울 48K 75T
material|M-0209|알루미늄 코일 A3003 3T
material|M-0210|스테인리스 코일 SUS304 1.2T
material|M-0211|스테인리스 코일 SUS316 1.2T
material|M-0212|실란트 변성실리콘 회색
material|M-0213|셀프 드릴링 스크류 5x50
material|M-0301|SC마닐라지 300g
material|M-0302|SC마닐라지 350g
material|M-0303|아이보리지 300g
material|M-0304|골판지 원지 라이너 KLB 180g
material|M-0305|골판지 원지 골심지 120g
material|M-0306|옵셋 잉크 CMYK 먹
material|M-0307|옵셋 잉크 CMYK 청
material|M-0308|UV 코팅액 유광
material|M-0309|UV 코팅액 무광
material|M-0310|핫멜트 접착제
material|M-0311|OPP 필름 20mic 유광
material|M-0312|OPP 필름 20mic 무광
material|M-0313|PE 손잡이 끈 흑색
"""

# 형식: "표기=>정답코드". 정답: 코드 / NONE(마스터에 없음) / AMBIG:코드1,코드2(여럿이 가능 → 확정 금지)
# 범주:
#   code_variant  코드 표기만 다름 (대소문자, 하이픈, 선행 0, O/0 혼동)
#   name_exact    등록 품명 그대로 (확인용)
#   name_spacing  띄어쓰기·기호만 다름
#   name_reorder  어순·약어·괄호
#   name_synonym  동의어·영문·현장 용어
#   spec_pick     규격이 형제 품목을 가르는 경우 (정답 있음)
#   spec_trap     규격이 마스터에 없는 근접 품목 (정답 NONE: 비슷한 품목으로 매핑하면 오류)
#   ambiguous     규격이 빠져 후보가 여럿 (정답 AMBIG: 확정 금지)
#   conflict      코드와 품명이 서로 다른 품목을 가리킴 (PoC M13 유형, 정답 AMBIG)
#   unregistered  마스터에 없는 전혀 다른 품목 (정답 NONE)
VALUE_HAND = {
    "code_variant": """p-1001=>P-1001|P1001=>P-1001|P-01002=>P-1002|p1004=>P-1004|MP2008=>MP-2008|mp-2010=>MP-2010|MP 2003=>MP-2003
        |pk-3001=>PK-3001|PK3OO2=>PK-3002|PK_3009=>PK-3009|M0210=>M-0210|m-0101=>M-0101|M-101=>M-0101|M.0204=>M-0204
        |M-O3O6=>M-0306|m0312=>M-0312|P-1O13=>P-1013|MP-02011=>MP-2011""",
    "name_exact": """백상지 80g A4 500매=>P-1001|샌드위치 패널 EPS 50T 벽체용=>MP-2001|골판지 상자 A형 300x200x150=>PK-3001
        |표백 화학펄프 활엽수 BHKP=>M-0101|컬러강판 0.5T 백색=>M-0201|SC마닐라지 300g=>M-0301|옵셋 잉크 CMYK 먹=>M-0306
        |징크 패널 0.7T 돌출이음=>MP-2011""",
    "name_spacing": """백상지80gA4 500매=>P-1001|샌드위치패널 EPS 50T 벽체용=>MP-2001|스테인리스패널 SUS304 1.2T 헤어라인=>MP-2008
        |골판지상자 A형 300×200×150=>PK-3001|크라프트지120g 롤1000mm=>P-1008|컬러강판0.5T백색=>M-0201|EPS단열재 50T 비드법2종=>M-0204
        |OPP필름 20mic 유광=>M-0311|라벨스티커 원형50mm 무광=>PK-3009|화장지원단 2겹 30m=>P-1010|SC 마닐라지 350g=>M-0302
        |골판지원지 라이너 KLB180g=>M-0304|메탈패널 알루미늄3T 불소도장=>MP-2006|쇼핑백크라프트 대=>PK-3007""",
    "name_reorder": """A4 백상지 80g (500매)=>P-1001|EPS 샌드위치 패널 벽체용 50T=>MP-2001|SUS304 헤어라인 스테인리스 패널 1.2T=>MP-2008
        |A형 골판지 상자 (300x200x150)=>PK-3001|국전 아트지 150g=>P-1005|롤 크라프트지 1000mm 120g=>P-1008|백색 컬러강판 0.5T=>M-0201
        |활엽수 표백 화학펄프(BHKP)=>M-0101|비드법 2종 EPS 단열재 75T=>M-0205|유광 UV 코팅액=>M-0308|무광 OPP 필름 20mic=>M-0312
        |3단 접지 리플렛 A4=>PK-3010|중철 카탈로그 16p A4=>PK-3011|알루미늄 타공 패널 2T=>MP-2012|6구 펄프몰드 포장 완충재=>PK-3012
        |SUS316 스테인리스 코일 1.2T=>M-0211""",
    "name_synonym": """A4 copy paper 75g 2500 sheets box=>P-1004|Sandwich panel PU 50T wall=>MP-2004|stainless panel 316 mirror 1.2t=>MP-2010
        |kraft shopping bag L=>PK-3007|판넬 EPS 100T 지붕=>MP-2003|스뎅 판넬 304 HL 1.2T=>MP-2008|BHKP 펄프=>M-0101
        |침엽수 펄프 BSKP=>M-0102|OCC 고지=>M-0103|신문 폐지=>M-0104|칼라강판 백색 0.5mm=>M-0201|PU 폴리올=>M-0206
        |유리섬유 단열재 48K 75mm=>M-0208|먹 잉크 옵셋=>M-0306|cyan offset ink=>M-0307|무광 UV 바니시=>M-0309
        |OPP 무광필름 20㎛=>M-0312|GI 메탈 패널 1.2mm=>MP-2007|경질우레탄 샌드위치 판넬 50mm 벽=>MP-2004|스노우 화이트지 150g 국전=>P-1007
        |두루마리 화장지 원단 3ply=>P-1011|핫멜트 본드=>M-0310|직결 피스 5x50=>M-0213|쇼핑백 손잡이 끈 PE 검정=>M-0313""",
    "spec_pick": """백상지 100g A4=>P-1002|백상지 A3 80g=>P-1003|샌드위치 패널 EPS 75T=>MP-2002|SUS304 1.5T 헤어라인 패널=>MP-2009
        |골판지 상자 A형 400x300x200=>PK-3002|골판지 상자 B형=>PK-3003|라벨 스티커 50mm 무광=>PK-3009|컬러강판 0.6T 백색=>M-0203
        |컬러강판 은회색 0.5T=>M-0202|SC마닐라 350g=>M-0302|EPS 75T 단열재=>M-0205|아트지 200g=>P-1006|화장지 원단 3겹=>P-1011
        |크라프트지 80g 롤=>P-1009|컬러박스 화장품 대=>PK-3005|쇼핑백 크라프트 중=>PK-3006|우레탄 원액 MDI=>M-0207
        |스테인리스 코일 SUS316=>M-0211|옵셋 잉크 청=>M-0307|골판지 원단 DW=>P-1013""",
    "spec_trap": """백상지 90g A4 500매|샌드위치 패널 EPS 125T 벽체용|SUS304 2.0T 헤어라인 패널|골판지 상자 A형 500x400x300
        |라벨 스티커 원형 80mm 유광|컬러강판 0.8T 백색|SC마닐라지 400g|아트지 250g 국전|크라프트지 100g 롤 1000mm
        |EPS 단열재 100T 비드법 2종|스테인리스 코일 SUS430 1.2T|OPP 필름 30mic 유광|화장지 원단 4겹 30m|글라스울 64K 75T
        |징크 패널 1.0T 돌출이음|골판지 상자 C형 300x200x150|옵셋 잉크 CMYK 황|알루미늄 코일 A5052 3T|복사용지 80g A4 2500매 박스
        |샌드위치 패널 우레탄 75T 벽체용""",
    "ambiguous": """백상지 A4=>AMBIG:P-1001,P-1002|샌드위치 패널 EPS=>AMBIG:MP-2001,MP-2002,MP-2003|스테인리스 패널 SUS304=>AMBIG:MP-2008,MP-2009
        |골판지 상자 A형=>AMBIG:PK-3001,PK-3002|쇼핑백 크라프트=>AMBIG:PK-3006,PK-3007|라벨 스티커 원형 50mm=>AMBIG:PK-3008,PK-3009
        |컬러강판 0.5T=>AMBIG:M-0201,M-0202|UV 코팅액=>AMBIG:M-0308,M-0309|옵셋 잉크=>AMBIG:M-0306,M-0307|SC마닐라지=>AMBIG:M-0301,M-0302
        |우레탄 원액=>AMBIG:M-0206,M-0207|OPP 필름 20mic=>AMBIG:M-0311,M-0312|아트지 국전=>AMBIG:P-1005,P-1006
        |화장지 원단 30m=>AMBIG:P-1010,P-1011|컬러박스 화장품용=>AMBIG:PK-3004,PK-3005|폐지=>AMBIG:M-0103,M-0104""",
    "conflict": """MP-2008 스테인리스 패널 SUS316 1.2T 미러=>AMBIG:MP-2008,MP-2010|P-1001 백상지 100g A4 500매=>AMBIG:P-1001,P-1002
        |PK-3001 골판지 상자 B형 300x200x150=>AMBIG:PK-3001,PK-3003|M-0201 컬러강판 0.6T 백색=>AMBIG:M-0201,M-0203
        |M-0308 UV 코팅액 무광=>AMBIG:M-0308,M-0309|MP-2001 샌드위치 패널 우레탄 50T 벽체용=>AMBIG:MP-2001,MP-2004
        |M-0306 옵셋 잉크 CMYK 청=>AMBIG:M-0306,M-0307|PK-3006 쇼핑백 크라프트 대=>AMBIG:PK-3006,PK-3007""",
    "unregistered": """PVC 타일 3T|합판 12T|에폭시 도료 회색|면장갑|PP 밴드|LED 조명 모듈|베어링 6204|유압 오일 46|택배 송장 용지
        |목재 팔레트 1100|Q-9001|X-0001|ZZ-123|동파이프 15A|방청유|작업복 상의""",
}


def build_master() -> list[dict]:
    rows = []
    for line in MASTER.strip().splitlines():
        kind, code, name = line.split("|")
        rows.append({"kind": kind, "code": code, "name": name})
    assert len({r["code"] for r in rows}) == len(rows)
    return rows


def build_value_queries(master: list[dict]) -> list[dict]:
    codes = {r["code"] for r in master}
    by_vkey: dict[str, list[str]] = {}
    by_name: dict[str, list[str]] = {}
    for r in master:
        by_vkey.setdefault(_variant_key(r["code"]), []).append(r["code"])
        by_name.setdefault(norm_name(r["name"]), []).append(r["code"])
    rows, seen = [], set()
    for category, text in VALUE_HAND.items():
        for item in text.replace("\n", "").split("|"):
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            query, _, gold = item.partition("=>")
            query, gold = query.strip(), (gold.strip() or "NONE")
            accept = ""
            if gold.startswith("AMBIG:"):
                accept, gold = gold[6:], "AMBIG"
                assert all(c in codes for c in accept.split(",")), item
            elif gold != "NONE":
                assert gold in codes, item
            # 현재 백엔드 방식의 기준선: 코드 정확 일치 → 선행 0·대소문자만 다른 코드(_variant_key) → 품명 정규화 일치(norm_name)
            if query in codes:
                rule = query
            else:
                hit = by_vkey.get(_variant_key(query), []) or by_name.get(norm_name(query), [])
                rule = hit[0] if len(hit) == 1 else ""
            rows.append({"qid": f"B{len(rows) + 1:04d}", "query": query, "gold": gold, "accept": accept, "category": category,
                         "rule_pred": rule, "split": split_of(f"B|{query}")})
    return rows


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    fields = build_fields()
    (OUT / "fields.json").write_text(json.dumps(fields, ensure_ascii=False, indent=1), encoding="utf-8")
    hq = build_header_queries()
    write_csv(OUT / "header_queries.csv", hq)
    master = build_master()
    write_csv(OUT / "master_items.csv", master)
    vq = build_value_queries(master)
    write_csv(OUT / "value_queries.csv", vq)

    def count(rows):
        c: dict[str, dict[str, int]] = {}
        for r in rows:
            c.setdefault(r["category"], {"dev": 0, "test": 0})[r["split"]] += 1
        return c

    manifest = {"dataset": "mapping-eval-v1", "synthetic": True,
                "header_queries": {"n": len(hq), "by_category": count(hq)},
                "master_items": len(master),
                "value_queries": {"n": len(vq), "by_category": count(vq)},
                "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(OUT.glob("*")) if p.suffix in (".csv", ".json") and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("header_queries", "master_items", "value_queries")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
