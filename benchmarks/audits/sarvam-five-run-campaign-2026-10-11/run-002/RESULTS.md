# Run002 — schema accepted, comparison incomplete

26calls dispatched,25resolved,1invalidanswer,1truncatedextraction; all usage known. Eight completed extractions (three empty/five nonempty),9validcontrolanswers and7validSQLiteanswers. No headline accuracy:16validanswers do not complete the48intended. No retry.

Concrete issues: D01-hi SQLite answer copied pickup_counter_id from augmentation rather than required pickup_counter; fact K7 itself was correct. D07-te extraction returned escaped malformed output and repetitive whitespace, finish_reason=length,4096completiontokens,0reasoningtokens. No claim this proves provider grammar caused it; it is a reason to test a declared alternate contract path.

Before run003: move the same extraction schema to system-prompt text and use documented JSON-object mode. Local validation/rejection remains unchanged. All answer requests, model/settings, sources/gold/schedule/budgets remain fixed. New independent slot; historical run remains stopped. Fact meaning is reviewed separately; no product/architecture win.
