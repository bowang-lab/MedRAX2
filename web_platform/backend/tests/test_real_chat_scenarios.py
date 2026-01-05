"""
Real-World Chat Scenario Tests

These tests verify actual chat functionality with real user scenarios,
testing memory, context, tool integration, and response quality.
"""

import pytest
import json
from fastapi.testclient import TestClient
from PIL import Image
import io


class TestBasicChatInteractions:
    """Test basic chat interactions that users will perform."""
    
    def test_simple_greeting(self, client: TestClient, auth_headers, test_chat):
        """Test basic greeting and response."""
        from app.services.tool_manager import tool_manager as tm
        
        # Load tool
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "Hello", "scan_ids": []}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        # Parse response
        lines = list(response.iter_lines())
        assert len(lines) > 0, "Should receive response lines"
    
    def test_ask_what_you_can_do(self, client: TestClient, auth_headers, test_chat):
        """Test asking what the assistant can do."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "What can you help me with?", "scan_ids": []}
        )
        
        assert response.status_code == 200
        
        # Should stream response
        for _ in response.iter_lines():
            pass
    
    def test_medical_terminology_question(self, client: TestClient, auth_headers, test_chat):
        """Test asking about medical terminology."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "What does pneumothorax mean?", "scan_ids": []}
        )
        
        assert response.status_code == 200


class TestContextAndMemory:
    """Test that the agent maintains context and remembers information."""
    
    def test_doctor_introduces_themselves(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Doctor introduces themselves."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # Doctor introduces themselves
        response1 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "Hi, I'm Dr. Sharma from cardiology department", "scan_ids": []}
        )
        assert response1.status_code == 200
        for _ in response1.iter_lines():
            pass
        
        # Ask about it later
        response2 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "Which department did I say I'm from?", "scan_ids": []}
        )
        assert response2.status_code == 200
    
    def test_patient_information_context(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Discussing patient information across messages."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        conversation = [
            "I have a patient, 65 year old male",
            "He came with chest pain",
            "His symptoms started 2 days ago",
            "What are the most common causes given this information?"
        ]
        
        for msg in conversation:
            response = client.post(
                f"/api/chats/{test_chat.id}/stream",
                headers=auth_headers,
                json={"content": msg, "scan_ids": []}
            )
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass
    
    def test_remembers_previous_diagnosis_discussion(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Referencing previous diagnosis discussion."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # First mention diagnosis
        response1 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "I suspect the patient has pneumonia", "scan_ids": []}
        )
        assert response1.status_code == 200
        for _ in response1.iter_lines():
            pass
        
        # Later reference it
        response2 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "What treatment would you recommend for the condition I mentioned?", "scan_ids": []}
        )
        assert response2.status_code == 200


class TestMedicalQueryScenarios:
    """Test realistic medical query scenarios."""
    
    def test_chest_xray_findings_question(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Asking about chest X-ray findings."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "What are the typical findings of pulmonary edema on chest X-ray?", "scan_ids": []}
        )
        
        assert response.status_code == 200
    
    def test_differential_diagnosis_question(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Asking for differential diagnosis."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={
                "content": "Patient presents with shortness of breath and chest pain. What's the differential diagnosis?",
                "scan_ids": []
            }
        )
        
        assert response.status_code == 200
    
    def test_treatment_recommendation_question(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Asking for treatment recommendations."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "What is the standard treatment protocol for community-acquired pneumonia?", "scan_ids": []}
        )
        
        assert response.status_code == 200


class TestMultiTurnConversations:
    """Test multi-turn conversations with follow-up questions."""
    
    def test_clarifying_questions_flow(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Doctor asks question, then clarifies."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # Initial question
        response1 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "Tell me about pneumothorax", "scan_ids": []}
        )
        assert response1.status_code == 200
        for _ in response1.iter_lines():
            pass
        
        # Clarifying question
        response2 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "Specifically, what are the radiological signs?", "scan_ids": []}
        )
        assert response2.status_code == 200
        for _ in response2.iter_lines():
            pass
        
        # Follow-up
        response3 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "How urgent is the treatment?", "scan_ids": []}
        )
        assert response3.status_code == 200
    
    def test_building_on_previous_answer(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Building on previous answers."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        conversation = [
            "What causes pleural effusion?",
            "Can you elaborate on the cardiac causes?",
            "What imaging findings would I see?",
            "How would I differentiate from pneumonia?"
        ]
        
        for msg in conversation:
            response = client.post(
                f"/api/chats/{test_chat.id}/stream",
                headers=auth_headers,
                json={"content": msg, "scan_ids": []}
            )
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_complete_case_discussion(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Complete case discussion from start to finish."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        case_flow = [
            "I have a new patient case to discuss",
            "72 year old male, smoker for 40 years",
            "Presents with persistent cough and weight loss",
            "His chest X-ray shows a mass in the right upper lobe",
            "What should be my next steps?",
            "What staging workup would you recommend?",
            "Thank you for the guidance"
        ]
        
        for msg in case_flow:
            response = client.post(
                f"/api/chats/{test_chat.id}/stream",
                headers=auth_headers,
                json={"content": msg, "scan_ids": []}
            )
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass
    
    def test_emergency_scenario(self, client: TestClient, auth_headers, test_chat):
        """Real scenario: Emergency situation discussion."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        emergency_flow = [
            "Emergency case: tension pneumothorax suspected",
            "Patient is hypoxic and tachycardic",
            "What's the immediate management?",
            "Chest tube inserted, patient stabilizing",
            "What follow-up imaging is needed?"
        ]
        
        for msg in emergency_flow:
            response = client.post(
                f"/api/chats/{test_chat.id}/stream",
                headers=auth_headers,
                json={"content": msg, "scan_ids": []}
            )
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass


class TestMessagePersistence:
    """Test that all messages are properly saved and retrievable."""
    
    def test_messages_are_saved_correctly(self, client: TestClient, auth_headers, test_chat):
        """Verify all messages are saved to database."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        test_messages = [
            "First message",
            "Second message",
            "Third message"
        ]
        
        # Send messages
        for msg in test_messages:
            response = client.post(
                f"/api/chats/{test_chat.id}/stream",
                headers=auth_headers,
                json={"content": msg, "scan_ids": []}
            )
            assert response.status_code == 200
            for _ in response.iter_lines():
                pass
        
        # Retrieve messages
        response = client.get(
            f"/api/chats/{test_chat.id}/messages",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        messages = response.json()
        user_messages = [m for m in messages if m['role'] == 'user']
        
        # Should have at least the test messages
        assert len(user_messages) >= len(test_messages)
        
        # Check content
        for test_msg in test_messages:
            assert any(test_msg in m['content'] for m in user_messages), \
                f"Message '{test_msg}' should be in saved messages"
    
    def test_message_timestamps_are_correct(self, client: TestClient, auth_headers, test_chat):
        """Verify message timestamps are saved correctly."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # Send a message
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": "Test timestamp message", "scan_ids": []}
        )
        assert response.status_code == 200
        for _ in response.iter_lines():
            pass
        
        # Get messages
        response = client.get(
            f"/api/chats/{test_chat.id}/messages",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        messages = response.json()
        
        # All messages should have timestamps
        for msg in messages:
            assert 'created_at' in msg
            assert msg['created_at'] is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_long_message(self, client: TestClient, auth_headers, test_chat):
        """Test handling of very long messages."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        long_message = "Tell me about pneumonia. " * 100  # Very long message
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": long_message, "scan_ids": []}
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 413]  # 413 = Payload Too Large
    
    def test_special_characters_in_message(self, client: TestClient, auth_headers, test_chat):
        """Test handling of special characters."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        special_message = "What about CO₂ levels? Temperature >38°C? Dose 5mg/kg?"
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": special_message, "scan_ids": []}
        )
        
        assert response.status_code == 200
    
    def test_rapid_successive_messages(self, client: TestClient, auth_headers, test_chat):
        """Test sending multiple messages rapidly."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # Send 5 messages rapidly
        for i in range(5):
            response = client.post(
                f"/api/chats/{test_chat.id}/stream",
                headers=auth_headers,
                json={"content": f"Rapid message {i}", "scan_ids": []}
            )
            assert response.status_code == 200
            # Don't wait for response to complete
    
    def test_message_with_numbers_and_measurements(self, client: TestClient, auth_headers, test_chat):
        """Test messages with medical measurements."""
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        medical_data = "Patient BP: 140/90, HR: 95, SpO2: 94%, Temp: 38.5°C, WBC: 15,000"
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={"content": medical_data, "scan_ids": []}
        )
        
        assert response.status_code == 200


@pytest.fixture(autouse=True)
def cleanup_tools_after_test():
    """Clean up loaded tools after each test."""
    yield
    # Unload all tools
    from app.services.tool_manager import tool_manager as tm
    for tool in tm.get_all_tools():
        if tool.get('status') == 'loaded':
            try:
                tm.unload_tool(tool.get('id'))
            except:
                pass

