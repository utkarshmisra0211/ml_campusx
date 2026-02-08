from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class DivisibleBy5(BaseNode[None, None, int]):
    foo: int

    async def run(self, ctx: GraphRunContext) -> Union["Increment", End[int]]:
        if self.foo % 5 == 0:
            return End(self.foo)
        return Increment(self.foo)


@dataclass
class Increment(BaseNode):
    foo: int

    async def run(self, ctx: GraphRunContext) -> DivisibleBy5:
        return DivisibleBy5(self.foo + 1)


fives_graph = Graph(nodes=[DivisibleBy5, Increment])


async def main() -> None:
    result = await fives_graph.run(DivisibleBy5(4))
    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
