"""Test tool factory functions with mock dependencies."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agent.shopping_tools import (
    create_search_products,
    create_get_product_detail,
    create_get_reviews,
    create_compare_products,
    create_get_user_profile,
    create_update_user_profile,
)


class TestCreateSearchProducts:
    def test_returns_formatted_result(self):
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = "## 产品\n### 1. Test"
        tool = create_search_products(mock_retriever)

        result = tool.invoke({"query": "游戏本", "top_k": 5})
        assert "Test" in result
        mock_retriever.retrieve.assert_called_once()

    def test_clamps_top_k(self):
        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = "ok"
        tool = create_search_products(mock_retriever)

        tool.invoke({"query": "test", "top_k": 999})
        _, kwargs = mock_retriever.retrieve.call_args
        assert kwargs["top_k"] <= 8

    def test_handles_none_retriever(self):
        tool = create_search_products(None)
        result = tool.invoke({"query": "test", "top_k": 5})
        assert "错误" in result or "未初始化" in result


class TestCreateGetProductDetail:
    def test_handles_missing_product(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE products (product_id INTEGER, name TEXT, brand TEXT, category TEXT, subcategory TEXT, price INTEGER, specs TEXT, description TEXT, rating REAL, sales_count INTEGER, release_date TEXT)")
        conn.commit()
        conn.close()

        tool = create_get_product_detail(db_path)
        result = tool.invoke({"product_id": 999})
        assert "未找到" in result


class TestCreateUserProfile:
    def test_get_profile_handles_none_store(self):
        tool = create_get_user_profile(None)
        result = tool.invoke({"conv_id": "test"})
        assert "未初始化" in result

    def test_update_profile_handles_none_store(self):
        tool = create_update_user_profile(None)
        result = tool.invoke({"conv_id": "test", "key": "budget", "value": "8000"})
        assert "未初始化" in result


class TestCreateCompareProducts:
    def test_rejects_single_product(self, tmp_path):
        tool = create_compare_products(str(tmp_path / "test.db"))
        result = tool.invoke({"product_ids": "1"})
        assert "至少" in result
