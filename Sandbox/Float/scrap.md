
```scala
trait Term{ type B; val x:B }

def f(t:Term): t.B = ...

class StringTerm(val foo:String) extends Term { B=String; x=foo}
class IntTerm(val baz:Int) extends Term { B=Int; x=baz}
t= new StringTerms("five")
s=new IntTerm(5)
u:String = f(t)
v:Int = f(s)
```
