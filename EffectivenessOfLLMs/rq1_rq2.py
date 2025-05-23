import json
import os

from tests.test_control import database_name

from evaluation_for_DLBench import init_database
from langchain.chains.constitutional_ai.prompts import examples
from query_llm_chat import query_llm, Item
from dbms_connectors.connector_factory import get_connector_by_dbms_name
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file_path)

def load_few_shot_examples(db_):
    example_str = ""
    with open(os.path.join("..", "few-shot-examples", "bird-dlbench", db_+".json"), "r", encoding="utf-8") as r:
        examples = json.load(r)
    for index in range(len(examples)):
        example_str = example_str + f"[Example {index+1}:]" + str(examples[index])
    return example_str

def rq1_rq2_exp(dataset_type, input_file, output_file, db_, model_, stream_, temperature_, fs, ka):
    prompt_template_others = {
        "System messages": "You are an expert SQL translation assistant.Let's think step by step.",
        "Task Descriptions": "Please translate the following SQL statement from {source_dialect} to {target_dialect} , ensuring that the resulting query is functionally equivalent to the original. [SQL statement]:{source_query}\n",
        "Schema Information": "{source_dialect} schema: {source_related_schemas}.\n {target_dialect} schema: {target_related_schemas}.",
        "External Dialect Knowledge": "You may refer to these dialect rules and tips to guide your translation.{source_dialect} dialect knowledge:{source_dialect_knowledge}.\n {target_dialect} dialect knowledge: {target_dialect_knowledge}",
        "Few-shot Demonstrations": "[SQL] ... [Answer]",
        "Output Constrains": "Output only the translated SQL. Do not add any extra commentary."
    }

    prompt_template_deepseek_r1 = {
        "System messages": "You are an expert SQL translation assistant.Let's think step by step.",
        "Task Descriptions": "Please translate the following SQL statement from {source_dialect} to {target_dialect} , ensuring that the resulting query is functionally equivalent to the original. [SQL statement]:{source_query}\n",
        "Schema Information": "{source_dialect} schema: {source_related_schemas}.\n {target_dialect} schema: {target_related_schemas}.",
        "External Dialect Knowledge": "You may refer to these dialect rules and tips to guide your translation.{source_dialect} dialect knowledge:{source_dialect_knowledge}.\n {target_dialect} dialect knowledge: {target_dialect_knowledge}",
        "Few-shot Demonstrations": "[SQL] ... [Answer]",
        "Output Constrains": "Output only the translated SQL in the following JSON format:{{'answer': 'your answer SQL'}}.Do not add any extra commentary."
    }
    if model_ == "deepseek-r1:8b-llama-distill-q8_0":
        prompt_template = prompt_template_deepseek_r1
    else:
        prompt_template = prompt_template_others

    user_content = "[Task Descriptions]:" + prompt_template["Task Descriptions"] + "\n"+ prompt_template[
        "Output Constrains"] + "\n[Schema Information]:" + prompt_template["Schema Information"]

    if ka:
        user_content = user_content + "\n[External Dialect Knowledge]:" + prompt_template["External Dialect Knowledge"]

    with open(input_file, "r", encoding="utf-8") as r:
        data_load = json.load(r)
    length = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as r:
            result_load = r.readlines()
        # length = json.loads(result_load[-1])["sql_id"]
        length = len(result_load)

    for index in range(len(data_load)):
        item = data_load[index]
        # 判断是否已经处理过
        if index < length:
            continue
        print(item["sql_id"])

        params = {
            "source_dialect": item["source_dialect"],
            "target_dialect": item["target_dialect"],
            "source_query": item["source_query"],
            "source_related_schemas": item["source_related_schemas"],
            "target_related_schemas": item["target_related_schemas"],
            "source_dialect_knowledge": item["source_dialect_knowledge"],
            "target_dialect_knowledge": item["target_dialect_knowledge"],
        }
        user_format = user_content.format(**params)
        if fs:
            few_shot_demonstrations = load_few_shot_examples(db_)
            user_format = user_format + "\n[Few-shot Demonstrations]:" + few_shot_demonstrations
        formated_messages = [
            {"role": "system", "content": prompt_template["System messages"]},
            {"role": "user", "content": user_format}
        ]

        print(formated_messages)
        formated_item = Item(
            model=model_,
            messages=formated_messages,
            stream=stream_,
            temperature=temperature_
        )
        # print(formated_item.messages)
        response = query_llm(formated_item)
        response["sql_id"] = item["sql_id"]

        # 保存结果
        with open(output_file, "a", encoding="utf-8") as a:
            a.write(json.dumps(response, ensure_ascii=False) + "\n")

def bird_dlbench_rq1_rq2_exp():
    models = [
        "sqlcoder:7b",
        "codellama:7b-instruct",
        "deepseek-coder:6.7b-instruct",
        "deepseek-r1:8b-llama-distill-q8_0",
        "gpt-3.5-turbo"
    ]

    dbs = [
        "clickhouse",
        "duckdb",
        "mariadb",
        "monetdb",
        "mysql",
        "postgresql"
    ]

    models = [
        "codellama:7b-instruct",
        "deepseek-coder:6.7b-instruct",
        "deepseek-r1:8b-llama-distill-q8_0",
        "gpt-3.5-turbo"
    ]

    models = [
        "gpt-3.5-turbo"
    ]

    dbs = [
        "clickhouse",
        "duckdb",
        "mariadb",
        "monetdb",
        "mysql",
        "postgresql"
    ]

    #
    for db in dbs:
        input_file = "../bird-dlbench/bird-{db_name}.json".format(db_name=db)
        for model in models:
            output_file = "../bird-dlbench/bird-{db_name}-result-{model_name}.jsonl".format(db_name=db,
                                                                                            model_name=model)
            print(output_file)
            rq1_rq2_exp(1, input_file, output_file, db, model, False, 0.0, fs=False, ka=False)
            # rq1_rq2_exp(input_file, output_file, db, model, False, 0.0, fs=True, ka=False)
            # rq1_rq2_exp(input_file, output_file, model, False, 0.0, fs=False, ka=True)

def test_suites_extension_rq1_rq2_exp():
    models = [
        "sqlcoder:7b",
        "codellama:7b-instruct",
        "deepseek-coder:6.7b-instruct",
        "deepseek-r1:8b-llama-distill-q8_0",
        "gpt-3.5-turbo"
    ]

    dbs = [
        "clickhouse",
        "duckdb",
        "mariadb",
        "monetdb",
        "mysql",
        "postgresql"
    ]


    models = [
        "gpt-3.5-turbo"
    ]

    dbs = [
        "clickhouse",
        "duckdb",
        "mariadb",
        "monetdb",
        "mysql",
        "postgresql"
    ]

    # fo
    for db in dbs:
        input_file = os.path.join("..", "test-suites-extension", "dataset-processed", "{db_name}.json".format(db_name=db))
        output_dic = os.path.join("..", "test-suites-extension-Output", db)
        if not os.path.exists(output_dic):
            os.makedirs(output_dic)
        for model in models:
            output_file = os.path.join(output_dic, "test-suites-{db_name}-result-{model_name}.jsonl".format(db_name=db,model_name=model))
            print(output_file)
            rq1_rq2_exp(2,input_file, output_file, db, model, False, 0.0, fs=False, ka=False)
            # rq1_rq2_exp(input_file, output_file, db, model, False, 0.0, fs=True, ka=False)
            # rq1_rq2_exp(input_file, output_file, model, False, 0.0, fs=False, ka=True)

def test_suites_extension_process(dataset_name):
    dbs = [
        "clickhouse",
        "duckdb",
        "mariadb",
        "monetdb",
        "postgresql"
    ]

    # 处理test suites extension的数据，将每一组数据的ddl执行，再查询表格的schema，更新到数据中
    for db in dbs:
        input_dic = os.path.join("..",dataset_name,"dataset")
        output_dic = os.path.join("..", dataset_name, "dataset-processed")
        input_file = os.path.join(input_dic,"{db_name}.json".format(db_name=db))
        output_file = os.path.join(output_dic,"{db_name}.json".format(db_name=db))
        print(output_file)
        if os.path.exists(output_file):
            return
        if not os.path.exists(output_dic):
            os.makedirs(output_dic)
        with open(input_file, "r", encoding="utf-8") as r:
            data_load = json.load(r)

        for item in data_load:
            # monetdb暂时使用ddl代替，因为不懂这个db的语法
            if db in ["monetdb"]:
                target_database_name = item["database_name"]
                source_dialect = item["source_dialect"]  # "mysql"
                target_dialect = item["target_dialect"]  # "clickhouse"
                with open(os.path.join(current_dir, "..", dataset_name, "schemas", source_dialect, target_database_name + ".txt"),
                          "r",encoding="utf-8") as r:
                    item["source_related_schemas"] = r.readlines()
                with open(os.path.join(current_dir, "..", dataset_name, "schemas", target_dialect, target_database_name + ".txt"),
                          "r",
                          encoding="utf-8") as r:
                    item["target_related_schemas"] = r.readlines()
            else:
                target_database_name = item["database_name"]
                source_dialect = item["source_dialect"]  # "mysql"
                target_dialect = item["target_dialect"]  # "clickhouse"
                with open(os.path.join(current_dir, "..", dataset_name, "schemas-str", source_dialect, target_database_name + ".json"),
                          "r",encoding="utf-8") as r:
                    item["source_related_schemas"] = json.load(r)
                with open(os.path.join(current_dir, "..", dataset_name, "schemas-str", target_dialect, target_database_name + ".json"),
                          "r",encoding="utf-8") as r:
                    item["target_related_schemas"] = json.load(r)
        with open(output_file, "w", encoding="utf-8") as w:
            json.dump(data_load, w, indent=4)

def test_suites_extension_schema_to_str(dataset_name):
    # 处理test suites extension数据的内容

    show_schema_sql = {
        "clickhouse": '''
            SELECT 
                table AS TABLE_NAME,
                name AS COLUMN_NAME,
                type AS COLUMN_TYPE
            FROM system.columns
            WHERE database = '{database_name}'
            ORDER BY table, position
        ''',
        "duckdb": '''
            SELECT
                table_name AS TABLE_NAME,
                column_name AS COLUMN_NAME,
                data_type AS COLUMN_TYPE
            FROM information_schema.columns
            ORDER BY table_name, ordinal_position;
        ''',
        "mariadb":  ''' 
            SELECT 
                TABLE_NAME,
                COLUMN_NAME,
                COLUMN_TYPE,
                COLUMN_KEY  -- 添加主键信息（PRI 表示主键）
            FROM information_schema.columns
            WHERE table_schema = '{database_name}'
            ORDER BY TABLE_NAME, ORDINAL_POSITION;
        ''',
        "monetdb": '''
            SELECT 
                c.table_name,
                c.column_name,
                c."type" AS column_type,
                CASE 
                    WHEN pk.column_name IS NOT NULL THEN 'PRI'
                    ELSE ''
                END AS column_key
            FROM sys.columns c
            JOIN sys.tables t ON c.table_id = t.id
            JOIN sys.schemas s ON t.schema_id = s.id
            LEFT JOIN (
                SELECT 
                    k.table_id,
                    kc.name AS column_name
                FROM sys.keys k
                JOIN sys.columns kc ON k.column_id = kc.id
                WHERE k.type = 'primary'
            ) pk ON c.table_id = pk.table_id AND c.column_name = pk.column_name
            WHERE s.name = '{database_name}'
            ORDER BY t.name, c.column_number;

        ''',
        "mysql": ''' 
            SELECT 
                TABLE_NAME,
                COLUMN_NAME,
                COLUMN_TYPE,
                COLUMN_KEY  -- PRI 表示主键
            FROM information_schema.columns
            WHERE table_schema = '{database_name}'
            ORDER BY TABLE_NAME, ORDINAL_POSITION;
        ''',
        "postgresql":'''
            SELECT 
                cols.table_name,
                cols.column_name,
                pg_catalog.format_type(a.atttypid, a.atttypmod) AS column_type,
                CASE 
                    WHEN pk.constraint_type = 'PRIMARY KEY' THEN 'PRI'
                    ELSE ''
                END AS column_key
            FROM information_schema.columns cols
            JOIN pg_catalog.pg_attribute a
                ON a.attname = cols.column_name
                AND a.attrelid = (
                    SELECT c.oid 
                    FROM pg_catalog.pg_class c
                    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = cols.table_name AND n.nspname = cols.table_schema
                )
            LEFT JOIN (
                SELECT 
                    kcu.table_schema,
                    kcu.table_name,
                    kcu.column_name,
                    tc.constraint_type
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                    AND tc.table_name = kcu.table_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk 
                ON cols.table_schema = pk.table_schema
                AND cols.table_name = pk.table_name
                AND cols.column_name = pk.column_name
            WHERE cols.table_schema = 'public'
            ORDER BY cols.table_name, cols.ordinal_position;
        '''
    }


    dbs = [
        "clickhouse",
        "duckdb",
        "mariadb",
        "monetdb",
        "mysql",
        "postgresql"
    ]

    # 处理test suites extension的数据，将每一组数据的ddl执行，再查询表格的schema，更新到数据中
    for db in dbs:
        # monetdb暂时使用ddl代替，因为不懂这个db的语法
        if db in ["monetdb"]:
            continue
        input_dic = os.path.join("..","test-suites-extension","schemas",db)
        output_dic = os.path.join("..", "test-suites-extension", "schemas-str", db)
        if not os.path.exists(output_dic):
            os.makedirs(output_dic)
        input_files = os.listdir(input_dic)
        for input_file in input_files:
            output_file = os.path.join(output_dic, input_file.replace(".txt", ".json"))
            if os.path.exists(output_file):
                continue

            database_name = input_file.replace(".txt", "")

            # 获取ddl语句并初始化对应的database
            init_database("test-suites-extension", db, database_name)

            # 查询schemas
            with open(os.path.join(current_dir, "..", "dbms_connectors", "dbms_config.json"), "r", encoding="utf-8") as r:
                dbms_config = json.load(r)
            # 打开该db时，不清空该数据库（因为已经建表并初始化了）
            dbms_config[db]["drop_database"] = False
            if db == "duckdb":
                dbms_config[db]["db_path"] = os.path.join(current_dir, "..", dataset_name, "schemas",f"{database_name}.duckdb")
            else:
                dbms_config[db]["database_name"] = database_name
            connector = get_connector_by_dbms_name(db, **dbms_config[db])

            g_res, g_rowcount, g_error_message = connector.execute(show_schema_sql[db].format(database_name=database_name))

            # 将查询结果格式化为字符串
            schemas = {}
            for row in g_res:
                table_name = row[0]
                column_name = row[1]
                column_type = row[2]
                column_key = row[3] if len(row) >=4 else ""
                if table_name in schemas:
                    schemas[table_name].append([str(column_name), str(column_type), str(column_key)])
                else:
                    schemas[table_name] = []
                    schemas[table_name].append([str(column_name), str(column_type), str(column_key)])

            # 生成格式化字符串
            database_schema = []
            for table, columns in schemas.items():
                table_schema_str = f"Table: `{table}`\nColumns:\n"
                for column in columns:
                    column_name = column[0]
                    column_type = column[1]
                    column_key = column[2]
                    table_schema_str = table_schema_str + f"({column_name}, {column_type}, {column_key})" + "\n"
                # 将生成的table schema string合并到整体的
                database_schema.append(table_schema_str)

            # 将schema更新
            print(database_schema)
            with open(output_file, "w", encoding="utf-8") as w:
                json.dump(database_schema, w, indent=4)



if __name__ == "__main__":
    # test_suites_extension_process("test-suites-extension")
    # test_suites_extension_schema_to_str("test-suites-extension")

    test_suites_extension_rq1_rq2_exp()



