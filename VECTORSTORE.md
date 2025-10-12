# LangChain VectorStore API 지원 현황

이 문서는 LangChain의 VectorStore 인터페이스와 현재 envector에서 지원하는 기능들을 비교 분석한 결과입니다.

## API 지원 현황 테이블

| 메서드 | 설명 | 현재 상태 | 비고 |
|--------|------|-----------|------|
| **문서 추가/관리** |
| `add_documents(documents)` | Document 객체로 문서 추가 | 🔧 구현 가능 | `add_texts` 래핑으로 구현 가능 |
| `add_texts(texts, metadatas, ids)` | 텍스트로 직접 추가 | ✅ 구현됨 | 완전 지원 |
| `add_documents(documents)` | 문서 추가 | ✅ 지원 | `add_texts` 위임, 임베딩/벡터 경로 지원 |
| `upsert_documents(documents)` | 문서 추가/업데이트 | ❌ 구현 불가 | ES2 SDK 제한으로 불가능 |
| `upsert_texts(texts, metadatas, ids)` | 텍스트 추가/업데이트 | ❌ 구현 불가 | ES2 SDK 제한으로 불가능 |
| **문서 삭제** |
| `delete(ids)` | ID로 문서 삭제 | ❌ 구현 불가 | ES2 SDK 제한으로 불가능 |
| `delete_documents(documents)` | Document 객체로 삭제 | ❌ 구현 불가 | ES2 SDK 제한으로 불가능 |
| **검색** |
| `similarity_search(query, k, filter)` | 유사도 검색 | ✅ 구현됨 | 완전 지원 |
| `similarity_search_with_score(query, k, filter)` | 점수와 함께 유사도 검색 | 🔧 구현 가능 | `_score`를 메타데이터로 제공 중 |
| `similarity_search_by_vector(embedding, k, filter)` | 벡터로 직접 검색 | ✅ 구현됨 | 완전 지원 |
| `similarity_search_with_score_by_vector(embedding, k, filter)` | 벡터로 점수와 함께 검색 | 🔧 구현 가능 | `_score`를 메타데이터로 제공 중 |
| **팩토리 메서드** |
| `from_texts(texts, embedding, metadatas)` | 텍스트로부터 생성 | ✅ 구현됨 | 완전 지원 |
| `from_documents(documents, embedding)` | Document로부터 생성 | ✅ 구현됨 | 완전 지원 |
| **기타** |
| `as_retriever(**kwargs)` | VectorStoreRetriever로 변환 | ✅ 구현됨 | 완전 지원 |

### 범례
- ✅ **구현됨**: 현재 완전히 구현되어 사용 가능
- 🔧 **구현 가능**: 현재 구현되지 않았지만 기술적으로 구현 가능
- ❌ **구현 불가**: ES2 SDK 제한으로 인해 구현 불가능

## 지원 현황 요약

### ✅ 구현됨 (6개)
- `add_texts` - 텍스트 추가
- `similarity_search` - 유사도 검색
- `similarity_search_by_vector` - 벡터 검색
- `from_texts` - 팩토리 메서드
- `from_documents` - 팩토리 메서드
- `as_retriever` - 리트리버 변환

### 🔧 구현 가능 (3개)
- `add_documents` - Document 객체 추가 (래핑으로 구현 가능)
- `similarity_search_with_score` - 점수와 함께 검색 (현재 `_score` 메타데이터로 제공)
- `similarity_search_with_score_by_vector` - 벡터로 점수와 함께 검색 (현재 `_score` 메타데이터로 제공)

### ❌ 구현 불가 (4개)
- `add_documents` - Document 리스트 삽입 (지원)
- `upsert_documents` - 문서 업서트 (ES2 SDK 제한)
- `upsert_texts` - 텍스트 업서트 (ES2 SDK 제한)
- `delete` - ID로 삭제 (ES2 SDK 제한)
- `delete_documents` - Document 삭제 (ES2 SDK 제한)

## 주요 제한사항

1. **개별 문서 삭제/업데이트 불가**: envector는 개별 문서의 삭제나 업데이트를 지원하지 않습니다. 전체 인덱스를 삭제해야 합니다.
2. **IDs 무시**: `add_documents`/`add_texts`에서 사용자 제공 ID는 무시됩니다. 반환값은 서버 영속 ID가 아닌 일시적 식별자입니다.

2. **upsert 기능 없음**: 문서의 추가/업데이트를 한 번에 처리하는 upsert 기능이 없습니다.

3. **점수 반환 방식**: `similarity_search_with_score` 메서드는 없지만, `similarity_search`에서 `_score`를 메타데이터로 제공합니다.

## 사용 권장사항

- **문서 추가**: `add_texts` 메서드 사용
- **검색**: `similarity_search` 또는 `similarity_search_by_vector` 사용
- **점수 확인**: 검색 결과의 `metadata['_score']`에서 점수 확인
- **RAG 파이프라인**: `as_retriever()`를 사용하여 LangChain의 RAG 워크플로우에 통합

## 호환성

envector는 LangChain의 핵심 VectorStore 기능을 지원하여 기본적인 RAG(Retrieval-Augmented Generation) 워크플로우를 구현하는 데 충분합니다. 다만 개별 문서 관리가 필요한 경우에는 다른 VectorStore 구현체를 고려해야 합니다.
