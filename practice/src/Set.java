import java.util.*;
import java.util.function.Predicate;

public class Set {
    // Check Duplicate
    public static boolean hasDuplicate(int[] arr){
        HashSet<Integer>set = new HashSet<>();
        for(int num : arr){
            if(set.contains(num)){
                return true;
            }
            set.add(num);
        }
        return false;
    }

    // Map
    public static void frequency(int[] arr){
        HashMap<Integer,Integer>map = new HashMap<>();
        for(int num : arr){
            map.put(num,map.getOrDefault(num,0)+1);
        }
        System.out.println(map);
    }

    public static void main(String[] args){

        //HashSet
        HashSet<Integer>set = new HashSet<>();
        set.add(10);
        set.add(20);
        set.add(10); // ignored
        System.out.println(set);

        //HashMap
        HashMap<String,Integer>map = new HashMap<>();
        map.put("Bimal",23);
        System.out.println(map.get("Bimal"));

        // Predicate
        Predicate<Integer>isEven = x->x%2 == 0;
        System.out.println(isEven.test(10)); // true

        // Lambda Expression
        Runnable r = new Runnable() {
            public void run(){
                System.out.println("Running");
            }
        };

        // New Way (Lambda)
        Runnable r1 = ()-> System.out.println("Running");
    }
}