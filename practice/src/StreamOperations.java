import java.util.*;
import java.util.stream.Collectors;

public class StreamOperations {
    public static void main(String[] args){
        List<Integer>list = Arrays.asList(1,2,3,4);
        //Square All Numbers
        list.stream().map(x->x*x).forEach(System.out::println);

        //Count Even Numbers
        long count = list.stream().filter(x->x%2 ==0).count();
        System.out.println("Even no : "+count);

        List<Integer>even = list.stream().filter(x->x%2 == 0).collect(Collectors.toList());
        System.out.println("Even collections : "+even);
    }
}
