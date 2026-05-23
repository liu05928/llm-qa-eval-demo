from config import BASE_DIR, LOG_PATH, USE_MOCK
from llm_client import call_llm
from chat_logger import save_log
from prompt_templates import get_available_modes


def main():
    """
    命令行问答程序入口。
    """

    print("欢迎使用大模型问答 Demo")
    print(f"项目目录：{BASE_DIR}")

    if USE_MOCK:
        print("当前模式：模拟模式，不会调用真实 API")
    else:
        print("当前模式：真实 API 模式")

    print(f"支持的问答模式：{get_available_modes()}")
    print("输入 exit 可以退出程序")
    print("-" * 40)

    while True:
        question = input("请输入你的问题：").strip()

        if question.lower() == "exit":
            print("程序已退出。")
            break

        if not question:
            print("问题不能为空，请重新输入。")
            continue

        mode = input("请输入模式 general / education / paper_summary，直接回车默认为 general：").strip()

        if not mode:
            mode = "general"

        try:
            answer = call_llm(question, mode=mode)

            print("\n模型回答：")
            print(answer)

            save_log(question, answer, mode=mode)

            print(f"\n本轮问答已保存到：{LOG_PATH}")
            print("-" * 40)

        except Exception as e:
            print(f"\n程序运行出错：{e}")
            print("-" * 40)


if __name__ == "__main__":
    main()