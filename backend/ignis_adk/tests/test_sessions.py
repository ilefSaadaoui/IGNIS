import asyncio

from ignis.sessions import SessionManager


def test_session_create_resume_isolate_reset():
    async def scenario():
        sm = SessionManager()
        s1 = await sm.create_session(user_id="alice")
        s2 = await sm.create_session(user_id="bob")
        assert s1 != s2

        summ = await sm.load_summarizer("alice", s1)
        summ.set_question("alice's question")
        await sm.store_summary("alice", s1, summ)

        resumed = await sm.load_summarizer("alice", s1)
        assert resumed.summary.current_question == "alice's question"

        bob = await sm.load_summarizer("bob", s2)
        assert bob.summary.current_question == ""  # isolation

        reset_id = await sm.reset_session("alice", s1)
        assert reset_id == s1
        fresh = await sm.load_summarizer("alice", s1)
        assert fresh.summary.current_question == ""
        return True

    assert asyncio.run(scenario()) is True


def test_list_sessions_per_user():
    async def scenario():
        sm = SessionManager()
        await sm.create_session(user_id="carol")
        await sm.create_session(user_id="carol")
        listing = await sm.list_sessions("carol")
        return len(listing.sessions)

    assert asyncio.run(scenario()) == 2


def test_missing_session_raises_keyerror():
    async def scenario():
        sm = SessionManager()
        try:
            await sm.load_summarizer("nobody", "does-not-exist")
            return False
        except KeyError:
            return True

    assert asyncio.run(scenario()) is True
